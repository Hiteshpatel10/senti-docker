import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
import plotly.graph_objs as go
import plotly.express as px
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))
from wordcloud import WordCloud
import joblib
import emoji
import re
from collections import Counter

emoji_pattern = emoji.emojize(r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]+")


def extract_emojis(comment):
    return re.findall(emoji_pattern, comment)


def predict_sentiment(model, vectorizer, text):
    features = vectorizer.transform([text])
    features = features.reshape(1, -1)
    sentiment = model.predict(features)[0]
    return sentiment


def senti(uuid_str):
    df = pd.read_csv(f'./data/{uuid_str}.csv', parse_dates=['date'])
    df.dropna(subset=['comment'], inplace=True)
    filename = "./model/youtube/lr_model.joblib"
    model = joblib.load(filename)
    vectorizer = joblib.load("./model/youtube/vectorizer.joblib")

    df['sentiment'] = df['comment'].apply(lambda x: predict_sentiment(model, vectorizer, x))

    positive = df[df['sentiment'] == 1]
    negative = df[df['sentiment'] == -1]
    neutral = df[df['sentiment'] == 0]

    positive.to_csv(f'./data/{uuid_str}_positive.csv', index=False)
    negative.to_csv(f'./data/{uuid_str}_negative.csv', index=False)
    neutral.to_csv(f'./data/{uuid_str}_neutral.csv', index=False)


    counts = {'Negative': len(negative), 'Neutral': len(neutral), 'Positive': len(positive),}
    fig = px.bar(x=list(counts.keys()), y=list(counts.values()), color=list(counts.keys()))
    fig.update_traces(marker_line_width=1.5)
    fig.update_layout(title_text='Sentiment Analysis Results', xaxis_title='Sentiment Category', yaxis_title='Comment Count')
    fig.write_html(f'./static/youtube/sentiment_histogram_{uuid_str}.html')

    stopwords = set(STOPWORDS)
    stopwords.update(["br", "href", "good", "great"])

    if len(positive) > 0:
        pos = " ".join(str(review) for review in positive.comment) 
        wordcloud2 = WordCloud(stopwords=stopwords).generate(pos)
        plt.imshow(wordcloud2, interpolation='bilinear')
        plt.axis("off")
        plt.savefig(f'./static/youtube/wordcloud_positive_{uuid_str}.png', transparent=True)

    if len(negative) > 0:
        neg = " ".join(str(review) for review in negative.comment)
        wordcloud3 = WordCloud(stopwords=stopwords).generate(neg)
        plt.imshow(wordcloud3, interpolation='bilinear')
        plt.axis("off")
        plt.savefig(f'./static/youtube/wordcloud_negative_{uuid_str}.png', transparent=True)

    if len(neutral) > 0:
        neu = " ".join(str(review) for review in neutral.comment)
        wordcloud4 = WordCloud(stopwords=stopwords).generate(neu)
        plt.imshow(wordcloud4, interpolation='bilinear')
        plt.axis("off")
        plt.savefig(f'./static/youtube/wordcloud_neutral_{uuid_str}.png', transparent=True)


    pos_reviews_per_day = positive.groupby(positive['date'].dt.date)['comment'].count()
    neg_reviews_per_day = negative.groupby(negative['date'].dt.date)['comment'].count()
    neu_reviews_per_day = neutral.groupby(neutral['date'].dt.date)['comment'].count()

    reviews_per_day = df.groupby(df['date'].dt.date)['comment'].count()

    #Emoji Count

    df['emojis'] = df['comment'].apply(extract_emojis)
    emoji_counts = Counter(emoji for emojis in df['emojis'] for emoji in emojis)
    fig = go.Figure(go.Bar(x=list(emoji_counts.keys()), y=list(emoji_counts.values())))
    fig.update_layout(
        xaxis_tickangle=-90,
        xaxis_title="Emoji",
        yaxis_title="Frequency",
        title="Emoji Histogram"
    )
    fig.write_html(f'./static/youtube/emoji_histogram_{uuid_str}.html')


    # Create the time series plot for each sentiment
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pos_reviews_per_day.index, y=pos_reviews_per_day.values, name='Positive'))
    fig.add_trace(go.Scatter(x=neg_reviews_per_day.index, y=neg_reviews_per_day.values, name='Negative'))
    fig.add_trace(go.Scatter(x=neu_reviews_per_day.index, y=neu_reviews_per_day.values, name='Neutral'))
    fig.update_layout(title_text='Youtube Comment Time Series Plot (Sentiment)', xaxis_title='Date', yaxis_title='Number of Comments')
    # fig.write_image('./static/youtube/time_series_sentiment.png')
    fig.write_html(f'./static/youtube/time_series_sentiment_{uuid_str}.html')

    # Create the time series plot for reviews
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=reviews_per_day.index, y=reviews_per_day.values, name='Reviews'))
    fig.update_layout(title_text='Youtube Comment Time Series Plot (Comment)', xaxis_title='Date', yaxis_title='Number of Comments')
    # fig.write_image('./static/youtube/time_series.png')
    fig.write_html(f'./static/youtube/time_series_{uuid_str}.html')


if __name__ == "__main__":
    senti('abc')