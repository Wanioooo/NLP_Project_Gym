# ===============================
# PureGym Sentiment Analysis Dashboard
# ===============================

import streamlit as st
import pandas as pd
import plotly.express as px
from collections import Counter
from transformers import pipeline
from sklearn.metrics import confusion_matrix

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="PureGym Sentiment Dashboard",
    page_icon="🏋️",
    layout="wide"
)

st.title("🏋️ PureGym Customer Sentiment Analysis Dashboard")
st.write("Analyze customer reviews using AI-powered sentiment and emotion detection.")

# -------------------------------
# LABEL & EMOJI MAPS
# -------------------------------
label_map = {
    "LABEL_0": "negative",
    "LABEL_1": "neutral",
    "LABEL_2": "positive"
}

emoji_map = {
    "joy": "😄",
    "anger": "😡",
    "sadness": "😢",
    "fear": "😨",
    "surprise": "😲",
    "disgust": "🤢",
    "neutral": "😐"
}

# -------------------------------
# LOAD MODELS
# -------------------------------
@st.cache_resource
def load_models():
    sentiment_model = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment"
    )

    emotion_model = pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=5
    )

    return sentiment_model, emotion_model

sentiment_model, emotion_model = load_models()

# -------------------------------
# HELPER FUNCTIONS
# -------------------------------
def batch_predict(pipeline_model, texts, batch_size=8):
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        preds = pipeline_model(
            batch,
            truncation=True,
            max_length=256
        )
        results.extend(preds)
    return results

def rating_to_sentiment(rating):
    if rating <= 2:
        return "negative"
    elif rating == 3:
        return "neutral"
    else:
        return "positive"

# Optional: clean text if needed
def clean_text(text):
    return str(text).strip()

def fetch_tweets(query, limit=50):
    df = pd.read_csv("sample_tweets.csv")  # your collected tweets
    return df["tweet"].astype(str).tolist()[:limit]

# -------------------------------
# SINGLE REVIEW ANALYSIS
# -------------------------------
st.subheader("✍️ Single Review Analysis")

user_review = st.text_area("Enter your review:")

user_rating = st.radio(
    "Give a rating:",
    options=[1, 2, 3, 4, 5],
    format_func=lambda x: "⭐" * x,
    horizontal=True
)

if st.button("Analyze Review"):
    if user_review.strip():
        sentiment_result = sentiment_model(user_review)[0]
        sentiment_label = label_map[sentiment_result["label"]]
        sentiment_score = sentiment_result["score"]

        emotion_results = emotion_model(user_review)[0]
        emotion_dict = {e["label"]: round(e["score"] * 100, 2) for e in emotion_results}

        rating_sentiment = rating_to_sentiment(user_rating)

        st.subheader("🧠 Sentiment Result")
        st.write(f"**Predicted Sentiment:** {sentiment_label}")
        st.write(f"**Confidence:** {sentiment_score:.2f}")
        st.write(f"**Rating-based Sentiment:** {rating_sentiment}")

        st.subheader("🎭 Emotion Detection")
        df_emotion = pd.DataFrame({
            "Emotion": [f"{emoji_map.get(k,'')} {k.capitalize()}" for k in emotion_dict],
            "Score (%)": list(emotion_dict.values())
        }).sort_values("Score (%)")

        fig = px.bar(
            df_emotion,
            x="Score (%)",
            y="Emotion",
            orientation="h",
            text="Score (%)",
            title="Emotion Confidence (%)"
        )

        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(xaxis_range=[0, 100])

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("⚠️ Sentiment vs Rating Check")
        if sentiment_label != rating_sentiment:
            st.warning("Mismatch detected between AI sentiment and user rating!")
        else:
            st.success("Sentiment matches rating.")

# ======================================================
# 📁 BATCH REVIEW ANALYSIS
# ======================================================
st.header("📁 Batch Review Analysis (CSV Upload)")

uploaded_file = st.file_uploader("Upload a CSV file containing reviews:", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Preview of uploaded dataset:")
    st.dataframe(df.head())

    # Select review text column
    text_column = st.selectbox(
        "Select the column that contains review text:",
        df.columns
    )

    # Select rating column
    rating_column = st.selectbox(
        "Select the column that contains ratings (1–5):",
        df.columns
    )

    # Slider to limit rows for performance
    max_rows = st.slider(
        "Limit number of reviews to analyze (for performance):",
        min_value=100,
        max_value=min(2000, len(df)),
        value=500,
        step=100
    )

    if st.button("Analyze Dataset"):
        with st.spinner("Analyzing reviews... Please wait."):
            # Prepare text for prediction
            texts = df[text_column].astype(str).apply(clean_text).tolist()[:max_rows]

            # Predict AI sentiment
            sentiment_preds = batch_predict(sentiment_model, texts)

            # Create result dataframe
            df_result = df.head(max_rows).copy()
            df_result["ai_sentiment"] = [label_map[s["label"]] for s in sentiment_preds]
            df_result["ai_sentiment_score"] = [s["score"] for s in sentiment_preds]

            # Create rating sentiment
            df_result["rating_sentiment"] = df_result[rating_column].apply(rating_to_sentiment)

        st.success("Batch analysis completed!")
        st.dataframe(df_result.head())

        # -------------------------------
        # Batch Summary Statistics
        # -------------------------------
        st.subheader("📊 Batch Analysis Summary")
        total_reviews = len(df_result)
        sentiment_counts = df_result["ai_sentiment"].value_counts(normalize=True) * 100

        positive_pct = sentiment_counts.get("positive", 0)
        negative_pct = sentiment_counts.get("negative", 0)
        dominant_emotion = "N/A"  # Optional: can compute if emotion predictions are added

        st.write(f"*Total reviews analyzed:* {total_reviews}")
        st.write(f"*Positive reviews:* {positive_pct:.2f}%")
        st.write(f"*Negative reviews:* {negative_pct:.2f}%")
        st.write(f"*Dominant emotion:* {dominant_emotion}")

        if positive_pct > negative_pct:
            st.info("📌 Overall sentiment trend: Mostly Positive")
        else:
            st.info("📌 Overall sentiment trend: Mostly Negative")

        csv = df_result.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Results as CSV",
            csv,
            "sentiment_emotion_results.csv",
            "text/csv"
        )

        # -------------------------------
        # CONFUSION MATRIX
        # -------------------------------
        st.subheader("⚠️ Rating vs AI Sentiment Confusion Matrix")
        cm = confusion_matrix(
            df_result["rating_sentiment"],
            df_result["ai_sentiment"],
            labels=["negative", "neutral", "positive"]
        )

        cm_df = pd.DataFrame(
            cm,
            index=["Rating Negative", "Rating Neutral", "Rating Positive"],
            columns=["AI Negative", "AI Neutral", "AI Positive"]
        )

        st.dataframe(cm_df)

        mismatch_rate = (df_result["rating_sentiment"] != df_result["ai_sentiment"]).mean() * 100
        st.warning(f"{mismatch_rate:.1f}% of reviews show sentiment–rating mismatch")

else:
    st.info("⬆️ Upload a CSV file to begin analysis")

# ======================================================
# 🔴 LIVE TWITTER (X) SENTIMENT ANALYSIS
# ======================================================
st.header("🔴 Live Social Media Feed Analysis (Twitter/X)")

query = st.text_input(
    "Search keyword or hashtag:",
    value="PureGym"
)

tweet_limit = st.slider(
    "Number of tweets to analyze:",
    min_value=10,
    max_value=200,
    value=50,
    step=10
)

if st.button("Analyze Live Tweets"):
    with st.spinner("Fetching and analyzing tweets..."):
        tweets = fetch_tweets(query, tweet_limit)

        if len(tweets) == 0:
            st.warning("No tweets found.")
        else:
            sentiment_preds = batch_predict(sentiment_model, tweets)

            tweet_sentiments = [
                label_map[s["label"]] for s in sentiment_preds
            ]

            df_tweets = pd.DataFrame({
                "Tweet": tweets,
                "Predicted Sentiment": tweet_sentiments
            })

    st.success("Live Twitter analysis completed!")

    # -------------------------------
    # Sentiment Distribution
    # -------------------------------
    st.subheader("📊 Twitter Sentiment Distribution")

    sentiment_counts = Counter(tweet_sentiments)
    df_dist = pd.DataFrame({
        "Sentiment": sentiment_counts.keys(),
        "Count": sentiment_counts.values()
    })

    fig = px.pie(
        df_dist,
        names="Sentiment",
        values="Count",
        title="Live Twitter Sentiment Breakdown"
    )

    st.plotly_chart(fig, use_container_width=True)

    # -------------------------------
    # Display Tweets
    # -------------------------------
    st.subheader("📝 Recent Tweets & AI Sentiment")
    st.dataframe(df_tweets.head(20))

