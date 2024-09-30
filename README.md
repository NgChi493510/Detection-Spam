# **Spam Review Detection: Machine Learning and NLP Techniques**
## Overview
This repository contains the source code and data analysis used for detecting spam reviews on online shopping platforms. The increasing presence of spam reviews poses a significant threat to the reliability of consumer feedback, making the detection of such reviews critical for maintaining the trustworthiness of products and services. Our project applies various machine learning models combined with text extraction techniques to classify reviews as genuine or spam.



## Project Motivation
Online reviews have become a powerful tool for influencing consumer decisions, especially on e-commerce platforms. However, the rise of Opinion Spamming—where fake reviews are either automatically or manually generated—undermines the credibility of such systems. Our goal is to enhance the detection of these spam reviews through machine learning and Natural Language Processing (NLP) techniques. This project explores different text extraction methods such as Bag of Words (BoW), TF-IDF, and Word2Vec, and applies them to various machine learning models, including Logistic Regression, K-Nearest Neighbors (KNN), LightGBM, XGBoost, and CatBoost.

## Dataset
The dataset used for this project consists of 40,000 product reviews sourced from Amazon:

- 20,000 authentic reviews written by real users.
- 20,000 artificially generated reviews created using the GPT-2 language model.
These reviews cover multiple product categories and feature a balanced distribution of genuine and spam reviews. The dataset is publicly available and was used to train, validate, and test our models.

## Methodology
### Text Extraction Techniques
We explored several text extraction techniques to transform raw textual data into numerical features suitable for machine learning models:

- Bag of Words (BoW): Counts the frequency of each word, disregarding grammar and word order.
- Term Frequency-Inverse Document Frequency (TF-IDF): Evaluates the importance of words within a review, giving more weight to unique terms.
- Word2Vec: Converts words into vector representations, capturing semantic meanings.
- Hashing Vectorizer: Uses a hash function to transform the features into fixed-length indices.
## Machine Learning Models
We experimented with a wide range of models to evaluate their effectiveness in detecting spam reviews:

1. Logistic Regression (baseline)
2. K-Nearest Neighbors (KNN)
3. LightGBM
4. XGBoost
5. AdaBoost
6. CatBoost
Each model was tested with different text extraction techniques to find the optimal combination for spam detection.

## Key Findings
- TF-IDF combined with Logistic Regression produced the most stable results in Phase 1, where unigram and bigram n-grams proved to be the most effective.
- CatBoost, when combined with Word2Vec, showed the best performance among the advanced models, demonstrating high robustness across different text extraction methods.
- The presence of text length as a feature was found to have minimal impact on the model's performance.
## Results
The evaluation was performed using several metrics:

- Accuracy
- F1-Score
- ROC-AUC (Receiver Operating Characteristic - Area Under Curve)
The experiments showed that CatBoost consistently delivered the best results with the highest accuracy and stability across all phases.

## Conclusion
Our research highlights the effectiveness of combining machine learning with text extraction techniques to detect spam reviews. TF-IDF and CatBoost emerged as the best pair, offering robust performance across various n-gram configurations. Future work could explore the use of deep learning models and the detection of human-generated spam reviews to further improve the system.

## Future Work
- Investigate human-generated spam reviews to broaden the model’s capabilities.
- Apply neural networks and deep learning for enhanced feature extraction and classification.
- Expand the detection system to multilingual reviews and non-Latin scripts.
## Limitations
While our models performed well with synthetic reviews, real-world applications may present additional challenges, especially with reviews written in various languages. Our current study is also limited to English reviews.

## Usage
- Requirements
- Python 3.x
- Scikit-learn
- LightGBM
- XGBoost
- CatBoost
- Pandas
- Numpy
- Matplotlib