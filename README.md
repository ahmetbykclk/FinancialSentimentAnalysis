## Financial Sentiment Analysis

This is a Python-based Financial Sentiment Analysis project that uses Support Vector Machines (SVM) and TF-IDF vectorization to predict the sentiment (negative, neutral, or positive) of financial text data. The dataset is assumed to be in CSV format with a column "Sentence" containing the financial text data and a column "Sentiment" containing the corresponding sentiment labels.

## Table of Contents

- [Dataset](#dataset)
- [Requirements](#requirements)
- [How it Works](#how-it-works)
- [Usage](#usage)
 
## Dataset

The dataset file data.csv contains the financial text data and their corresponding sentiment labels. The CSV format should have two columns, "Sentence" and "Sentiment."

You can download my dataset directly from this link:

https://www.kaggle.com/datasets/sbhatti/financial-sentiment-analysis/download?datasetVersionNumber=4

## Requirements

To run this project, you need the following dependencies:

- Python 3.x
- pandas
- scikit-learn
- nltk (Natural Language Toolkit)

You can install the required packages using the following command:

pip install pandas scikit-learn nltk

Additionally, you need to download the 'punkt' tokenizer from nltk, which can be done using the following code:

import nltk

nltk.download('punkt')

## How it Works

The Financial Sentiment Analysis works as follows:

1- The dataset (CSV format) is loaded, containing the financial text data and corresponding sentiment labels.

2- The text data is preprocessed by converting it to lowercase and tokenizing it using the 'punkt' tokenizer from nltk.

3- The data is split into training and validation sets using an 80-20 split.

4- The TF-IDF vectorizer is initialized and fit to the training data, transforming the text data into numerical vectors.

5- A Linear Support Vector Machine (SVM) classifier is initialized and trained on the training set.

6- Predictions are made on the validation set, and accuracy along with a classification report is printed.

7- The top 10 words for each sentiment class (negative, neutral, and positive) are printed based on their feature importance (coefficients) from the SVM classifier.

8- The first 10 test values along with their real sentiment and predicted sentiment are printed.

## Usage

1- Clone the repository or download the FinancialSentimentAnalysis.py and data.csv files.

2- Make sure you have Python 3.x installed on your system.

3- Install the required dependencies by running pip install pandas scikit-learn nltk.

4- Run the FinancialSentimentAnalysis.py script.

The script will load the dataset, preprocess the text data, train the SVM classifier using TF-IDF vectorization, and evaluate its performance on the validation set. Additionally, it will print the top 10 words for each sentiment class and show the sentiment predictions for the first 10 test values.
