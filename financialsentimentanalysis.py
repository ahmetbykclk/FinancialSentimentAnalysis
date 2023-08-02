import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score
import nltk
from nltk.tokenize import word_tokenize

# Download the 'punkt' tokenizer
nltk.download('punkt')

# Load the data from data.csv
data = pd.read_csv('data.csv')

# Preprocess the text data
data['Sentence'] = data['Sentence'].apply(lambda x: ' '.join(word_tokenize(x.lower())))

# Split the data into training and validation sets
train_data = data.sample(frac=0.8, random_state=42)
val_data = data.drop(train_data.index)

# Initialize the TF-IDF vectorizer and fit it to the training data
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=10000)  # You can experiment with n-gram range and max_features
X_train = tfidf_vectorizer.fit_transform(train_data['Sentence'])
X_val = tfidf_vectorizer.transform(val_data['Sentence'])

# Initialize and train the SVM classifier
svm_classifier = LinearSVC(C=1.0)  # You can experiment with different values of C (regularization parameter)
svm_classifier.fit(X_train, train_data['Sentiment'])

# Make predictions on the validation set
val_predictions = svm_classifier.predict(X_val)

# Calculate validation accuracy and print the classification report
val_accuracy = accuracy_score(val_data['Sentiment'], val_predictions)
print("Validation Accuracy:", val_accuracy)
print(classification_report(val_data['Sentiment'], val_predictions))

# Get the feature importance (coefficients) of the classifier
feature_importance = svm_classifier.coef_

# Get the feature names (words) from the TF-IDF vectorizer
feature_names = np.array(tfidf_vectorizer.get_feature_names_out())

# Print the top 10 words for each sentiment class
for i, sentiment in enumerate(["negative", "neutral", "positive"]):
    top_words_indices = np.argsort(feature_importance[i])[-10:]
    top_words = feature_names[top_words_indices]
    print(f"Top 10 words for {sentiment} sentiment: {', '.join(top_words)}")

# Write the first 10 test values, their real sentiment, and predicted sentiment to the output
test_data = data.head(10)
X_test = tfidf_vectorizer.transform(test_data['Sentence'])
test_predictions = svm_classifier.predict(X_test)

print("\nFirst 10 Test Values, Real Sentiment, and Predicted Sentiment:")
for i in range(len(test_data)):
    print(f"Text: {test_data['Sentence'].iloc[i]} - Real Sentiment: {test_data['Sentiment'].iloc[i]} - Predicted Sentiment: {test_predictions[i]}")
