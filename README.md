# Fake Product Review Detection Using Data Mining Techniques

This is a college Data Mining Techniques (DMT) project that classifies an Amazon product review as **Genuine** or **Fake** using Data Mining and Machine Learning techniques.

## Problem Statement
Online platforms suffer from fake reviews that mislead consumers and damage trust. Detecting deceptive opinion spam is a challenging task because fake reviews can be carefully crafted to look authentic. This project applies classical machine learning classifiers to determine whether a given review is genuine or fake based on its content and associated metadata.

## Objective
To build, evaluate, and demonstrate a robust data mining pipeline that preprocesses raw Amazon reviews, extracts relevant text and categorical features, trains classification algorithms (Multinomial Naive Bayes and Logistic Regression), and exposes the model via a simple, interactive user interface.

## Dataset

The project uses an Amazon Reviews Dataset.
- **File:** `dataset/amazon_reviews.txt`
- **Size:** 21,000 labeled reviews (10,500 Fake, 10,500 Genuine).

| Raw label | Meaning | Encoded class |
| --- | --- | --- |
| `__label2__` | Genuine review | 0 |
| `__label1__` | Fake review | 1 |

## Features Used

We extract multiple features rather than relying purely on text:
1. **Review Text:** `REVIEW_TEXT`
2. **Rating:** `RATING` (1 to 5)
3. **Verified Purchase:** `VERIFIED_PURCHASE` (Y or N)
4. **Product Category:** `PRODUCT_CATEGORY` (e.g. Wireless, Books, Beauty)

## Data Preprocessing
1. Clean text: convert to lowercase, strip HTML/URLs, expand contractions, and remove generic stopwords while keeping negations (e.g., 'not', 'never').
2. Handle missing data by providing appropriate defaults (e.g., unknown category, missing rating).
3. Scale continuous variables (`RATING`) using `MinMaxScaler`.
4. Encode categorical variables (`VERIFIED_PURCHASE`, `PRODUCT_CATEGORY`) using `OneHotEncoder`.
5. Extract textual patterns using `TfidfVectorizer` (Term Frequency-Inverse Document Frequency) restricted to a 5,000-word vocabulary with unigrams and bigrams.
6. The entire preprocessing pipeline is bundled into a `ColumnTransformer` to ensure no data leakage between train and test sets, and to ensure identical transformations during live inference.

## Data Mining Algorithms
The project trains and compares two robust classical algorithms:
1. **Multinomial Naive Bayes**: A probabilistic classifier based on Bayes' theorem, well-suited for text and sparse feature data.
2. **Logistic Regression**: A linear model that estimates the probability of a binary response, providing excellent calibration for confidence scores.

## Train/Test Methodology
- **Split:** 80% Training Data, 20% Test Data.
- **Stratification:** The 50/50 class balance is preserved in both sets.
- **Leakage Prevention:** Preprocessors (TF-IDF, Scaler, Encoder) are fitted **only** on the training set, then used to transform the test set.

## Evaluation Metrics
Models are evaluated on the untouched test set using:
- **Accuracy:** Overall correctness.
- **Precision:** Correctness of positive predictions.
- **Recall:** Ability to find all actual positives.
- **F1-score:** Harmonic mean of Precision and Recall.
- **Confusion Matrix:** Visualization of True Positives, True Negatives, False Positives, and False Negatives.

## How to train the model

```bash
pip install -r requirements.txt
python train_model.py
```
This script will parse the dataset, clean the reviews, split the data, train both models, calculate metrics, and save all artifacts (preprocessor, models, metrics) to the `models/` directory.

## How to run the application

```bash
streamlit run app.py
```
This will launch a clean, single-page web interface (with no sidebar) at `http://localhost:8501`. 

## Expected Application Workflow
1. Launch the app.
2. Enter the review text, title, rating, verified purchase status, and product category into the fields.
3. Click "Analyze Review".
4. The system loads the pre-trained `Logistic Regression` model, applies exactly the same transformations used during training, and outputs a **Prediction** (Genuine or Fake) along with a **Confidence %**.
5. The model comparison metrics (Accuracy, F1, Confusion Matrices) are displayed at the bottom of the page for transparency.

## Limitations
- The model's vocabulary is limited to the top 5,000 n-grams from the training data. Highly domain-specific jargon might be ignored.
- Sarcasm and deeply contextual deception are difficult for simple TF-IDF representations to capture.
- The model heavily relies on the dataset's specific distribution of "Verified Purchase". (In this dataset, fake reviews correlate strongly with unverified purchases).
