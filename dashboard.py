import pandas as pd
import nltk
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from nltk.sentiment import SentimentIntensityAnalyzer

# Download necessary NLP data
nltk.download('vader_lexicon')

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("AllReviewsAndSentimentCombined.csv")
    df = df.dropna(subset=['review'])  # Remove missing values
    df['review'] = df['review'].astype(str)  # Ensure text format
    df['sentiment'] = df['sentiment'].str.lower().str.strip()  # Normalize sentiment labels
    return df

df = load_data()

# Train ML Model
X_train, X_test, y_train, y_test = train_test_split(df['review'], df['sentiment'], test_size=0.2, random_state=42)

@st.cache_resource
def train_model():
    vectorizer = TfidfVectorizer()
    model = make_pipeline(vectorizer, MultinomialNB())
    model.fit(X_train, y_train)
    return model

model = train_model()

sia = SentimentIntensityAnalyzer()  # Initialize VADER

# Streamlit App UI
st.title("Sentiment Analysis Dashboard")

# User Input for Analysis
user_input = st.text_area("Enter text to analyze:")

if user_input:
    # ML Model Prediction
    model_prediction = model.predict([user_input])[0]

    # VADER Scores
    vader_scores = sia.polarity_scores(user_input)
    vader_sentiment = "Positive 😊" if vader_scores['compound'] >= 0.05 else "Negative 😞" if vader_scores['compound'] <= -0.05 else "Neutral 😐"

    # Subjectivity Score
    subjectivity = TextBlob(user_input).sentiment.subjectivity
    subjectivity_label = "Subjective" if subjectivity > 0.5 else "Objective"

    # Display Results
    st.write("### Model Prediction: ", model_prediction)
    st.write("### VADER Prediction: ", vader_sentiment)
    st.json(vader_scores)
    st.write("### Subjectivity: ", f"{subjectivity_label} ({subjectivity:.2f})")

    # Word Cloud
    st.subheader("Word Cloud of Input Text")
    wordcloud = WordCloud(width=500, height=300, background_color="white").generate(user_input)
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig)

# Sentiment Distribution Pie Chart
st.subheader("Sentiment Distribution in Dataset")
fig, ax = plt.subplots()
df['sentiment'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, colors=['green', 'gray', 'red'], ax=ax)
ax.set_ylabel('')
st.pyplot(fig)


#not working
# Subjectivity vs. Objectivity Pie Chart
# st.subheader("Subjectivity vs. Objectivity in Dataset")
# df['subjectivity'] = df['review'].apply(lambda x: TextBlob(x).sentiment.subjectivity)
# fig, ax = plt.subplots()
# df['subjectivity'].apply(lambda x: 'Subjective' if x > 0.5 else 'Objective').value_counts().plot.pie(
#     autopct='%1.1f%%', startangle=90, colors=['blue', 'orange'], ax=ax)
# ax.set_ylabel('')
# st.pyplot(fig)

# # Sentiment Trend Over Time (if timestamp column exists)
# if 'date' in df.columns:
#     st.subheader("Sentiment Trend Over Time")
#     df['date'] = pd.to_datetime(df['date'], errors='coerce')
#     df.dropna(subset=['date'], inplace=True)
#     df['date'] = df['date'].dt.to_period('M')  # Group by Month
#     sentiment_trend = df.groupby(['date', 'sentiment']).size().unstack().fillna(0)
    
#     fig, ax = plt.subplots(figsize=(10, 5))
#     sentiment_trend.plot(kind='line', ax=ax)
#     ax.set_title("Sentiment Trend Over Time")
#     ax.set_ylabel("Count")
#     st.pyplot(fig)
