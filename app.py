import re
from collections import Counter

import pandas as pd
import streamlit as st
from nltk.corpus import stopwords
from textblob import TextBlob
import nltk

# Download stopwords at runtime (first run will download once)
nltk.download("stopwords", quiet=True)

st.title("Social Risk Simulator for Renewable Energy Projects")

# --- Sidebar parameters -------------------------------------------------------
st.sidebar.header("Project Parameters")
project_name = st.sidebar.text_input("Project name")
technology_type = st.sidebar.selectbox(
    "Technology type", ["Solar", "Wind", "Hydro", "Geothermal", "Other"]
)
project_location = st.sidebar.text_input("Project location")
engagement_level = st.sidebar.selectbox(
    "Engagement level", ["Low", "Medium", "High"]
)

# --- File upload --------------------------------------------------------------
uploaded = st.file_uploader(
    "Upload meeting minutes (text or CSV)", type=["txt", "csv"]
)


def load_text(file):
    """Return plain text from uploaded file."""
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
        return " ".join(df.astype(str).fillna("").values.flatten())
    return file.read().decode("utf-8")


def analyze_sentiment(text):
    """Return list of sentiment labels for each sentence."""
    blob = TextBlob(text)
    labels = []
    for sentence in blob.sentences:
        polarity = sentence.sentiment.polarity
        if polarity > 0.05:
            labels.append("positive")
        elif polarity < -0.05:
            labels.append("negative")
        else:
            labels.append("neutral")
    return labels


def extract_keywords(text):
    """Return most frequent keywords after stopword removal."""
    words = re.findall(r"\b\w+\b", text.lower())
    stop = set(stopwords.words("english"))
    keywords = [w for w in words if w not in stop and len(w) > 3]
    return Counter(keywords).most_common(10)


if uploaded:
    raw_text = load_text(uploaded)
    st.subheader("Analysis Results")

    # Sentiment
    sentiment_labels = analyze_sentiment(raw_text)
    sentiment_counts = Counter(sentiment_labels)
    st.write("### Sentiment Distribution")
    st.bar_chart(pd.Series(sentiment_counts))

    # Themes / keywords
    top_keywords = extract_keywords(raw_text)
    st.write("### Main Themes / Topics")
    st.write(pd.DataFrame(top_keywords, columns=["Keyword", "Count"]))

    # Key issues (reusing keywords list)
    st.write("### Key Issues Highlighted")
    st.write(", ".join([kw for kw, _ in top_keywords]))

    # Social Risk Level (simple heuristic)
    total = sum(sentiment_counts.values())
    neg_ratio = sentiment_counts.get("negative", 0) / total if total else 0
    if neg_ratio > 0.5 or sentiment_counts.get("negative", 0) > 10:
        risk_level = "High"
    elif neg_ratio > 0.2:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    st.write(f"### Overall Social Risk Level: **{risk_level}**")

else:
    st.info("Upload a file to analyze meeting minutes.")

