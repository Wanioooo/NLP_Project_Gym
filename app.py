# ===============================
# PureGym Sentiment Analysis Dashboard
# ===============================

import streamlit as st
import pandas as pd
import plotly.express as px
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
def rating_to_sentiment(r):
    if r >= 4:
        return "positive"
    elif r == 3:
        return "neutral"
    else:
        return "negative"

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
        emotion_dict = {
            e["label"]: round(e["score"] * 100, 2) for e in emotion_results
        }

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

# -------------------------------
# FILE UPLOAD
# -------------------------------
st.subheader("📂 Upload Review Dataset")
st.markdown("📌 **CSV must contain columns named exactly:** `Review` and `Rating`")

uploaded_file = st.file_uploader(
    "Upload CSV file",
    type="csv"
)

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if not {"Review", "Rating"}.issubset(df.columns):
        st.error("CSV must contain 'Review' and 'Rating' columns.")
        st.stop()

    st.subheader("📄 Data Preview")
    st.dataframe(df.head())

    with st.spinner("Running sentiment analysis..."):
        sentiments = sentiment_model(df["Review"].astype(str).tolist())
        df["sentiment"] = [label_map[s["label"]] for s in sentiments]

    with st.spinner("Detecting emotions..."):
        emotions = emotion_model(df["Review"].astype(str).tolist())
        df["emotion"] = [e[0]["label"] for e in emotions]  # top emotion only

    df["rating_sentiment"] = df["Rating"].apply(rating_to_sentiment)

    # -------------------------------
    # KPIs
    # -------------------------------
    st.subheader("📊 Key Metrics")

    col1, col2, col3 = st.columns(3)

    col1.metric("Positive (%)", round((df["sentiment"] == "positive").mean() * 100, 1))
    col2.metric("Negative (%)", round((df["sentiment"] == "negative").mean() * 100, 1))
    col3.metric("Top Emotion", df["emotion"].value_counts().idxmax())

    # -------------------------------
    # VISUALS
    # -------------------------------
    st.subheader("📈 Sentiment Distribution")

    fig_sent = px.bar(
        df["sentiment"].value_counts().reset_index(),
        x="index",
        y="sentiment",
        labels={"index": "Sentiment", "sentiment": "Count"}
    )

    st.plotly_chart(fig_sent, use_container_width=True)

    st.subheader("🎭 Emotion Distribution")

    fig_emo = px.bar(
        df["emotion"].value_counts().reset_index(),
        x="index",
        y="emotion",
        labels={"index": "Emotion", "emotion": "Count"}
    )

    st.plotly_chart(fig_emo, use_container_width=True)

    # -------------------------------
    # CONFUSION MATRIX
    # -------------------------------
    st.subheader("⚠️ Rating vs AI Sentiment Confusion Matrix")

    cm = confusion_matrix(
        df["rating_sentiment"],
        df["sentiment"],
        labels=["negative", "neutral", "positive"]
    )

    cm_df = pd.DataFrame(
        cm,
        index=["Rating Negative", "Rating Neutral", "Rating Positive"],
        columns=["AI Negative", "AI Neutral", "AI Positive"]
    )

    st.dataframe(cm_df)

    mismatch_rate = (df["rating_sentiment"] != df["sentiment"]).mean() * 100
    st.warning(f"{mismatch_rate:.1f}% of reviews show sentiment–rating mismatch")

    st.subheader("✅ Final Annotated Dataset")
    st.dataframe(df)

else:
    st.info("⬆️ Upload a CSV file to begin analysis")
