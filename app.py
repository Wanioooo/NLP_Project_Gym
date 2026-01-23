# ===============================
# PureGym Sentiment Analysis Dashboard
# ===============================

import streamlit as st
import pandas as pd
import plotly.express as px
from transformers import pipeline
from sklearn.metrics import confusion_matrix
import numpy as np

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
# LOAD MODELS (HEAVY BUT POWERFUL)
# -------------------------------
@st.cache_resource
def load_models():
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment"
    )
    emotion_pipeline = pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=1
    )
    return sentiment_pipeline, emotion_pipeline

sentiment_model, emotion_model = load_models()

# -------------------------------
# FILE UPLOAD
# -------------------------------
uploaded_file = st.file_uploader(
    "📂 Upload PureGym Reviews CSV (must include: review_text, rating)",
    type="csv"
)

if uploaded_file is not None:

    # -------------------------------
    # READ DATA
    # -------------------------------
    df = pd.read_csv(uploaded_file)

    if "review_text" not in df.columns or "rating" not in df.columns:
        st.error("❌ CSV must contain 'review_text' and 'rating' columns.")
        st.stop()

    st.subheader("📄 Uploaded Data Preview")
    st.dataframe(df.head())

    # -------------------------------
    # SENTIMENT ANALYSIS
    # -------------------------------
    with st.spinner("🔍 Analyzing sentiment..."):
        sentiment_results = sentiment_model(df["review_text"].astype(str).tolist())
        df["sentiment"] = [s["label"] for s in sentiment_results]

    # -------------------------------
    # EMOTION ANALYSIS
    # -------------------------------
    with st.spinner("😊 Detecting emotions..."):
        emotion_results = emotion_model(df["review_text"].astype(str).tolist())
        df["emotion"] = [e[0]["label"] for e in emotion_results]

    # -------------------------------
    # KPI METRICS
    # -------------------------------
    st.subheader("📊 Key Insights")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Positive Reviews (%)",
        round((df["sentiment"] == "LABEL_2").mean() * 100, 1)
    )
    col2.metric(
        "Negative Reviews (%)",
        round((df["sentiment"] == "LABEL_0").mean() * 100, 1)
    )
    col3.metric(
        "Most Common Emotion",
        df["emotion"].value_counts().idxmax()
    )

    # -------------------------------
    # SENTIMENT DISTRIBUTION
    # -------------------------------
    st.subheader("📈 Sentiment Distribution")

    sentiment_map = {
        "LABEL_0": "Negative",
        "LABEL_1": "Neutral",
        "LABEL_2": "Positive"
    }

    df["sentiment_label"] = df["sentiment"].map(sentiment_map)

    fig_sentiment = px.bar(
        df["sentiment_label"].value_counts().reset_index(),
        x="index",
        y="sentiment_label",
        labels={"index": "Sentiment", "sentiment_label": "Count"},
        title="Sentiment Polarity"
    )
    st.plotly_chart(fig_sentiment, use_container_width=True)

    # -------------------------------
    # EMOTION DISTRIBUTION
    # -------------------------------
    st.subheader("🎭 Emotion Distribution")

    fig_emotion = px.bar(
        df["emotion"].value_counts().reset_index(),
        x="index",
        y="emotion",
        labels={"index": "Emotion", "emotion": "Count"},
        title="Detected Emotions"
    )
    st.plotly_chart(fig_emotion, use_container_width=True)

    # -------------------------------
    # RATING TO SENTIMENT MAPPING
    # -------------------------------
    def rating_to_sentiment(r):
        if r <= 2:
            return "Negative"
        elif r == 3:
            return "Neutral"
        else:
            return "Positive"

    df["rating_sentiment"] = df["rating"].apply(rating_to_sentiment)

    # -------------------------------
    # CONFUSION MATRIX
    # -------------------------------
    st.subheader("⚠️ Rating vs AI Sentiment Confusion")

    cm = confusion_matrix(
        df["rating_sentiment"],
        df["sentiment_label"],
        labels=["Negative", "Neutral", "Positive"]
    )

    cm_df = pd.DataFrame(
        cm,
        index=["Rating Negative", "Rating Neutral", "Rating Positive"],
        columns=["AI Negative", "AI Neutral", "AI Positive"]
    )

    st.dataframe(cm_df)

    mismatch_rate = (df["rating_sentiment"] != df["sentiment_label"]).mean() * 100

    st.warning(
        f"⚠️ {round(mismatch_rate, 1)}% of reviews have mismatched rating and sentiment."
    )

    # -------------------------------
    # LIVE SOCIAL MEDIA FEED (SIMULATED)
    # -------------------------------
    st.subheader("📡 Live Social Media Feed (Simulated)")

    live_feed = [
        "PureGym is packed tonight 😡",
        "Love the new equipment at PureGym!",
        "Staff were helpful but gym was noisy",
        "Best gym experience so far 💪",
        "Too crowded, hard to workout properly"
    ]

    live_sentiment = sentiment_model(live_feed)
    live_emotion = emotion_model(live_feed)

    live_df = pd.DataFrame({
        "Post": live_feed,
        "Sentiment": [sentiment_map[s["label"]] for s in live_sentiment],
        "Emotion": [e[0]["label"] for e in live_emotion]
    })

    st.dataframe(live_df)

    # -------------------------------
    # FINAL TABLE
    # -------------------------------
    st.subheader("✅ Final Annotated Dataset")
    st.dataframe(df)

else:
    st.info("⬆️ Please upload a CSV file to start analysis.")

# --- Single Review Analysis ---
st.subheader("Single Review Analysis")
user_review = st.text_area("Enter your review:")

# Star rating input (horizontal stars)
user_rating = st.radio(
    "Rate the restaurant:",
    options=[1, 2, 3, 4, 5],
    format_func=lambda x: "⭐" * x,
    horizontal=True
)

if st.button("Analyze Review"):
    if user_review.strip() != "":
        # --- Sentiment prediction ---
        sentiment_result = sentiment_pipeline(user_review)[0]
        sentiment_label = label_map.get(sentiment_result['label'], sentiment_result['label'])
        sentiment_score = sentiment_result['score']

        # --- Emotion prediction ---
        emotion_results = emotion_pipeline(user_review)[0]
        emotion_dict = {e['label'].lower(): e['score'] for e in emotion_results}

        # --- Map rating to sentiment ---
        def rating_to_sentiment(rating):
            if rating >= 4:
                return "positive"
            elif rating == 3:
                return "neutral"
            else:
                return "negative"

        rating_sentiment = rating_to_sentiment(user_rating)

        # --- Display results ---
        st.subheader("Sentiment Analysis")
        st.write(f"**Sentiment:** {sentiment_label}")
        st.write(f"**Confidence:** {sentiment_score:.2f}")
        st.write(f"**Rating Sentiment:** {rating_sentiment}")

        # --- Emotion Pie Chart ---
        st.subheader("Emotion Analysis (Bar Chart)")

        # Prepare DataFrame
        df_emotion = pd.DataFrame({
            "Emotion": [f"{emoji_map.get(k, '')} {k.capitalize()}" for k in emotion_dict.keys()],
            "Score": [round(v*100, 2) for v in emotion_dict.values()]  # convert to percentage
        })
       
        # Sort for better visual
        df_emotion = df_emotion.sort_values("Score", ascending=True)
       
        # Plot colorful horizontal bar chart
        fig = px.bar(
            df_emotion,
            x="Score",
            y="Emotion",
            orientation="h",
            text="Score",
            color="Score",
            color_continuous_scale="Viridis",  # you can use other scales like "Rainbow", "Plasma"
            title="Emotion Confidence (%)"
        )
       
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(xaxis_title="Confidence (%)", yaxis_title="", xaxis_range=[0, 100])
       
        st.plotly_chart(fig)


        # --- Compare sentiment and rating ---
        st.subheader("Sentiment vs Rating Check")
        if sentiment_label != rating_sentiment:
            st.warning("⚠️ Mismatch detected!")
            st.dataframe(pd.DataFrame([{
                "Review": user_review,
                "Rating": "⭐" * user_rating,
                "Rating Sentiment": rating_sentiment,
                "Predicted Sentiment": sentiment_label,
                "Confidence": sentiment_score
            }]))
        else:
            st.success("✅ No mismatch detected.")
