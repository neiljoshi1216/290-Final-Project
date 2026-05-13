from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATA_FILE = Path(__file__).with_name("Amazon_Reviews.csv")
OUTPUT_FILE = Path(__file__).with_name("Amazon_Reviews_Cleaned.csv")
CONFUSION_MATRIX_PLOT = Path(__file__).with_name("logistic_confusion_matrix_heatmap.png")
SENTIMENT_BAR_PLOT = Path(__file__).with_name("sentiment_label_bar_chart.png")
SUSPICIOUS_PIE_PLOT = Path(__file__).with_name("suspicious_flag_pie_chart.png")
SENTIMENT_RATING_SCATTER_PLOT = Path(__file__).with_name("sentiment_vs_rating_scatter.png")


def ensure_nltk_resource(resource_path: str, download_name: str) -> None:
    # Download NLTK resources only if they are not already available locally.
    try:
        nltk.data.find(resource_path)
    except LookupError:
        nltk.download(download_name, quiet=True)


def normalize_text(value: object, stop_words: set[str], lemmatizer: WordNetLemmatizer) -> str:
    # Clean text for NLP by lowercasing, removing non-letters, filtering stopwords, and lemmatizing.
    if pd.isna(value):
        return ""

    text = str(value).lower().strip()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = [token for token in text.split() if token and token not in stop_words]
    lemmas = [lemmatizer.lemmatize(token) for token in tokens]
    return " ".join(lemmas)


def label_sentiment(compound_score: float) -> str:
    # Convert the VADER compound score into the standard sentiment buckets.
    if compound_score >= 0.05:
        return "Positive"
    if compound_score <= -0.05:
        return "Negative"
    return "Neutral"


def build_near_duplicate_flags(df: pd.DataFrame, similarity_threshold: float = 0.9) -> pd.Series:
    # Flag very similar review texts from different accounts using TF-IDF and nearest-neighbor cosine similarity.
    if len(df) < 2:
        return pd.Series(False, index=df.index)

    clean_text = df["Clean Review Text"].fillna("")
    valid_mask = clean_text.str.len() > 0
    flags = pd.Series(False, index=df.index)
    if valid_mask.sum() < 2:
        return flags

    tfidf = TfidfVectorizer(min_df=2, ngram_range=(1, 2))
    text_matrix = tfidf.fit_transform(clean_text[valid_mask])
    if text_matrix.shape[0] < 2:
        return flags

    neighbor_count = min(3, text_matrix.shape[0])
    neighbors = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=neighbor_count)
    neighbors.fit(text_matrix)
    distances, indices = neighbors.kneighbors(text_matrix)

    valid_indices = df.index[valid_mask]
    account_keys = df.loc[valid_indices, "Account Key"]
    for row_position, row_index in enumerate(valid_indices):
        current_account = account_keys.loc[row_index]
        for neighbor_position in range(1, neighbor_count):
            neighbor_row_position = indices[row_position, neighbor_position]
            neighbor_index = valid_indices[neighbor_row_position]
            neighbor_account = account_keys.loc[neighbor_index]
            similarity = 1 - distances[row_position, neighbor_position]
            if neighbor_account != current_account and similarity >= similarity_threshold:
                flags.loc[row_index] = True
                break

    return flags


def print_classification_metrics(y_true: pd.Series, y_pred: pd.Series) -> None:
    # Report the requested evaluation metrics for the suspicious-review classifier.
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"Recall: {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"F1 Score: {f1_score(y_true, y_pred, zero_division=0):.4f}")


def save_confusion_matrix_heatmap(matrix: np.ndarray) -> None:
    # Save a heatmap-style confusion matrix for the logistic regression output.
    figure, axis = plt.subplots(figsize=(6, 5))
    heatmap = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(heatmap, ax=axis)
    axis.set_title("Logistic Regression Confusion Matrix")
    axis.set_xlabel("Predicted Label")
    axis.set_ylabel("True Label")
    axis.set_xticks([0, 1], labels=["Not Suspicious", "Suspicious"])
    axis.set_yticks([0, 1], labels=["Not Suspicious", "Suspicious"])

    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            axis.text(
                column_index,
                row_index,
                f"{matrix[row_index, column_index]}",
                ha="center",
                va="center",
                color="black",
            )

    figure.tight_layout()
    figure.savefig(CONFUSION_MATRIX_PLOT, dpi=300, bbox_inches="tight")
    plt.close(figure)


def save_sentiment_bar_chart(df: pd.DataFrame) -> None:
    # Save a bar chart showing the distribution of Positive, Neutral, and Negative labels.
    sentiment_counts = df["Sentiment_Label"].value_counts().reindex(["Positive", "Neutral", "Negative"], fill_value=0)

    figure, axis = plt.subplots(figsize=(7, 5))
    axis.bar(sentiment_counts.index, sentiment_counts.values, color=["#4CAF50", "#9E9E9E", "#E53935"])
    axis.set_title("Sentiment Label Distribution")
    axis.set_xlabel("Sentiment Label")
    axis.set_ylabel("Number of Reviews")

    for index, value in enumerate(sentiment_counts.values):
        axis.text(index, value, str(value), ha="center", va="bottom")

    figure.tight_layout()
    figure.savefig(SENTIMENT_BAR_PLOT, dpi=300, bbox_inches="tight")
    plt.close(figure)


def save_suspicious_pie_chart(df: pd.DataFrame) -> None:
    # Save a pie chart showing the percentage of reviews flagged as suspicious.
    suspicious_counts = df["Suspicious"].value_counts().reindex([0, 1], fill_value=0)
    labels = ["Not Flagged", "Flagged"]

    figure, axis = plt.subplots(figsize=(6, 6))
    axis.pie(
        suspicious_counts.values,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90,
        colors=["#42A5F5", "#FF7043"],
    )
    axis.set_title("Percentage of Reviews Flagged as Suspicious")

    figure.tight_layout()
    figure.savefig(SUSPICIOUS_PIE_PLOT, dpi=300, bbox_inches="tight")
    plt.close(figure)


def save_sentiment_rating_scatter(df: pd.DataFrame) -> None:
    # Save a scatter plot of sentiment score versus rating with a fitted regression line.
    scatter_df = df[["Sentiment_Compound", "Rating"]].dropna().copy()
    x_values = scatter_df["Sentiment_Compound"].to_numpy(dtype=float)
    y_values = scatter_df["Rating"].to_numpy(dtype=float)
    regression_coefficients = np.polyfit(x_values, y_values, 1)
    regression_line = np.poly1d(regression_coefficients)
    x_line = np.linspace(x_values.min(), x_values.max(), 200)

    figure, axis = plt.subplots(figsize=(7, 5))
    axis.scatter(x_values, y_values, alpha=0.25, color="#1E88E5", edgecolors="none")
    axis.plot(x_line, regression_line(x_line), color="#D81B60", linewidth=2)
    axis.set_title("Sentiment Score vs Rating")
    axis.set_xlabel("VADER Compound Sentiment Score")
    axis.set_ylabel("Rating")

    figure.tight_layout()
    figure.savefig(SENTIMENT_RATING_SCATTER_PLOT, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    # ====================
    # PHASE 1: DATA CLEANING AND PREPARATION
    # ====================

    # 1. Load the file with pandas and inspect the dataset using df.info() and df.describe().
    # STEP 1: Load the raw review file and inspect structure, column types, and summary statistics.
    print("STEP 1: Load file and inspect structure")
    df = pd.read_csv(DATA_FILE)

    print("\ndf.info()")
    df.info()
    print("\ndf.describe(include='all')")
    print(df.describe(include="all").transpose())

    # 2. Handle missing values in the core columns and fill optional columns with placeholders.
    # STEP 2: Check missing values and clean the core columns needed for analysis.
    print("\nSTEP 2: Handle missing values")
    print(df.isnull().sum())

    text_columns = ["Reviewer Name", "Profile Link", "Country", "Review Title", "Review Text"]
    for column in text_columns:
        if column in df.columns:
            # Strip extra whitespace from text fields before any downstream processing.
            df[column] = df[column].astype("string").str.strip()

    # Fill softer missing fields with a placeholder and drop rows missing the main review content.
    df["Country"] = df["Country"].fillna("Unknown")
    df["Profile Link"] = df["Profile Link"].fillna("Unknown")
    df = df.dropna(subset=["Review Text", "Rating"]).copy()
    df = df[df["Review Text"].str.len() > 0].copy()

    print("\nMissing values after handling:")
    print(df.isnull().sum())

    # 3. Remove duplicates, while keeping repeated review texts available for later suspicious-review checks.
    # STEP 3: Remove exact duplicate rows while preserving repeated review texts for suspicious-review signals.
    print("\nSTEP 3: Remove duplicates")
    rows_before = len(df)
    full_duplicate_count = df.duplicated().sum()
    duplicate_text_count = df.duplicated(subset=["Review Text"]).sum()
    df = df.drop_duplicates().copy()
    print(f"Rows before duplicate removal: {rows_before}")
    print(f"Exact duplicate rows found: {full_duplicate_count}")
    print(f"Duplicate Review Text rows found: {duplicate_text_count}")
    print(f"Rows after duplicate removal: {len(df)}")

    # 4. Standardize data types for dates, ratings, and review counts.
    # STEP 4: Convert dates and numeric-looking strings into analysis-ready data types.
    print("\nSTEP 4: Standardize data types")
    df["Review Date"] = pd.to_datetime(df["Review Date"], errors="coerce", utc=True)
    df["Date of Experience"] = pd.to_datetime(
        df["Date of Experience"],
        errors="coerce",
        format="%d-%b-%y",
    )
    df["Rating"] = (
        df["Rating"]
        .astype("string")
        .str.extract(r"(\d+)", expand=False)
        .astype("Int64")
    )
    df["Review Count"] = (
        df["Review Count"]
        .astype("string")
        .str.extract(r"(\d+)", expand=False)
        .astype("Int64")
    )
    # Keep only valid star ratings in the expected 1-5 range.
    df = df[df["Rating"].between(1, 5, inclusive="both")].copy()

    print(df[["Review Date", "Date of Experience", "Rating", "Review Count"]].dtypes)

    # 5. Text preprocessing for NLP on Review Title and Review Text.
    #    This covers lowercasing, removing punctuation/numbers, stopword removal, and lemmatization.
    # STEP 5: Prepare review text for NLP analysis using normalization, stopword removal, and lemmatization.
    print("\nSTEP 5: Text preprocessing for NLP")
    ensure_nltk_resource("corpora/stopwords", "stopwords")
    ensure_nltk_resource("corpora/wordnet", "wordnet")
    ensure_nltk_resource("corpora/omw-1.4", "omw-1.4")
    ensure_nltk_resource("sentiment/vader_lexicon", "vader_lexicon")

    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

    df["Clean Review Title"] = df["Review Title"].apply(
        normalize_text,
        args=(stop_words, lemmatizer),
    )
    df["Clean Review Text"] = df["Review Text"].apply(
        normalize_text,
        args=(stop_words, lemmatizer),
    )
    df["Account Key"] = df["Profile Link"].fillna("Unknown") + "|" + df["Reviewer Name"].fillna("Unknown")

    print(df[["Review Title", "Clean Review Title", "Review Text", "Clean Review Text"]].head(3))

    # 6. Feature engineering for review length, days gap, short-review flag, and high-review-count flag.
    # STEP 6: Create analysis features that can help identify suspicious or low-information reviews.
    print("\nSTEP 6: Feature engineering")
    df["Review Char Count"] = df["Review Text"].str.len()
    df["Review Word Count"] = df["Review Text"].str.split().str.len()

    df["Days Gap"] = (df["Review Date"].dt.tz_localize(None) - df["Date of Experience"]).dt.days

    df["Is Short Review"] = df["Review Word Count"] < 10

    review_count_threshold = df["Review Count"].quantile(0.95)
    df["High Review Count Flag"] = df["Review Count"] >= review_count_threshold

    print(f"High review count threshold (95th percentile): {review_count_threshold}")
    print(
        df[
            [
                "Review Char Count",
                "Review Word Count",
                "Days Gap",
                "Is Short Review",
                "High Review Count Flag",
            ]
        ].head()
    )

    # ====================
    # PHASE 2: ANALYTICS AND MODELING
    # ====================

    # A. Sentiment analysis with VADER.
    #    Create a compound sentiment score and classify each review as Positive, Neutral, or Negative.
    # STEP 7: Run VADER sentiment analysis and create a Positive/Neutral/Negative sentiment label.
    print("\nSTEP 7: Sentiment analysis with VADER")
    sentiment_analyzer = SentimentIntensityAnalyzer()
    df["Sentiment_Compound"] = df["Review Text"].apply(
        lambda value: sentiment_analyzer.polarity_scores(str(value))["compound"]
    )
    df["Sentiment_Label"] = df["Sentiment_Compound"].apply(label_sentiment)

    print(df[["Rating", "Sentiment_Compound", "Sentiment_Label"]].head())

    # B. Rule-based flagging system for suspicious reviews.
    #    Signals included:
    #    - Sentiment-Rating Mismatch
    #    - Extremely Short Review with an extreme rating
    #    - Large Days Gap
    #    - High Review Count / hundreds of reviews
    #    - Duplicate or Near-Duplicate Text across different accounts
    # STEP 8: Build rule-based suspicious-review flags from mismatch, brevity, timing, review volume, and text similarity.
    print("\nSTEP 8: Rule-based suspicious review flags")

    # B1. Sentiment-Rating Mismatch: positive text with a low rating, or negative text with a high rating.
    df["Sentiment_Rating_Mismatch_Flag"] = (
        ((df["Sentiment_Label"] == "Positive") & (df["Rating"] <= 2))
        | ((df["Sentiment_Label"] == "Negative") & (df["Rating"] >= 4))
    )

    # B2. Extremely Short Review: a very short review paired with an extreme 1-star or 5-star rating.
    df["Extremely_Short_Extreme_Rating_Flag"] = (
        (df["Review Word Count"] <= 8) & (df["Rating"].isin([1, 5]))
    )

    # B3. Large Days Gap: a review posted long before or after the stated date of experience.
    df["Large_Days_Gap_Flag"] = df["Days Gap"].abs() >= 30

    # B4. High Review Count: reviewer has hundreds of reviews, which can indicate bot-like behavior.
    df["Hundreds_Review_Count_Flag"] = df["Review Count"] >= 100

    # B5. Duplicate/Near-Duplicate Text: same or highly similar text appearing across different accounts.
    df["Duplicate_Text_Flag"] = df.duplicated(subset=["Clean Review Text"], keep=False) & (
        df.groupby("Clean Review Text")["Account Key"].transform("nunique") > 1
    )
    df["Near_Duplicate_Text_Flag"] = build_near_duplicate_flags(df)
    df["Suspicious"] = (
        df[
            [
                "Sentiment_Rating_Mismatch_Flag",
                "Extremely_Short_Extreme_Rating_Flag",
                "Large_Days_Gap_Flag",
                "Hundreds_Review_Count_Flag",
                "Duplicate_Text_Flag",
                "Near_Duplicate_Text_Flag",
            ]
        ]
        .any(axis=1)
        .astype(int)
    )

    print(
        df[
            [
                "Sentiment_Rating_Mismatch_Flag",
                "Extremely_Short_Extreme_Rating_Flag",
                "Large_Days_Gap_Flag",
                "Hundreds_Review_Count_Flag",
                "Duplicate_Text_Flag",
                "Near_Duplicate_Text_Flag",
                "Suspicious",
            ]
        ]
        .sum()
        .to_string()
    )

    # C. Regression analysis.
    #    1. Logistic regression predicts whether a review is Suspicious (0/1).
    #    2. Linear regression predicts Rating from sentiment and review length.
    #    3. Evaluation uses confusion matrix, accuracy, precision, recall, F1, and R^2.
    # STEP 9: Train regression models to predict suspicious reviews and rating alignment.
    print("\nSTEP 9: Regression analysis")

    # C1. Logistic regression features for predicting Suspicious reviews.
    logistic_features = df[
        ["Review Word Count", "Days Gap", "Review Count", "Sentiment_Compound", "Rating"]
    ].copy()
    logistic_features["Days Gap"] = logistic_features["Days Gap"].fillna(logistic_features["Days Gap"].median())
    logistic_features["Review Count"] = logistic_features["Review Count"].fillna(0)
    y_logistic = df["Suspicious"]

    X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(
        logistic_features,
        y_logistic,
        test_size=0.2,
        random_state=42,
        stratify=y_logistic,
    )

    logistic_model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )
    logistic_model.fit(X_train_log, y_train_log)
    logistic_predictions = logistic_model.predict(X_test_log)
    logistic_confusion = confusion_matrix(y_test_log, logistic_predictions)

    # C1 Evaluation: confusion matrix, accuracy, precision, recall, and F1 score.
    print("\nLogistic Regression Metrics")
    print_classification_metrics(y_test_log, logistic_predictions)

    # C2. Linear regression features for predicting Rating from sentiment score and review length.
    linear_features = df[["Sentiment_Compound", "Review Word Count"]].copy()
    y_linear = df["Rating"].astype(float)

    X_train_lin, X_test_lin, y_train_lin, y_test_lin = train_test_split(
        linear_features,
        y_linear,
        test_size=0.2,
        random_state=42,
    )

    linear_model = LinearRegression()
    linear_model.fit(X_train_lin, y_train_lin)
    linear_predictions = linear_model.predict(X_test_lin)

    # C2 Evaluation: R^2 score to measure how well sentiment and review length explain the rating.
    print("\nLinear Regression Metrics")
    print(f"R^2 Score: {r2_score(y_test_lin, linear_predictions):.4f}")

    # STEP 10: Save charts for the logistic model, sentiment mix, suspicious flag percentage, and sentiment-rating relationship.
    print("\nSTEP 10: Create charts")
    save_confusion_matrix_heatmap(logistic_confusion)
    save_sentiment_bar_chart(df)
    save_suspicious_pie_chart(df)
    save_sentiment_rating_scatter(df)
    print(f"Saved chart: {CONFUSION_MATRIX_PLOT}")
    print(f"Saved chart: {SENTIMENT_BAR_PLOT}")
    print(f"Saved chart: {SUSPICIOUS_PIE_PLOT}")
    print(f"Saved chart: {SENTIMENT_RATING_SCATTER_PLOT}")

    # Final output: save the enriched dataset with cleaning, sentiment, flags, and model-ready columns.
    # Save the cleaned dataset so it can be used in later modeling or visualization steps.
    print("\nSaving cleaned dataset")
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved cleaned file to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()