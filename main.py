import requests
import json
from flask import Flask, request
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from bs4 import BeautifulSoup
import nltk

nltk.download('punkt')
nltk.download('vader_lexicon')

app = Flask(__name__)

@app.route('/')
def index():
    return '''
        <form action="/process-noun" method="post">
            <label for="noun">Enter a noun:</label>
            <input type="text" id="noun" name="noun" required>
            <button type="submit">Submit</button>
        </form>
    '''

@app.route('/process-noun', methods=['POST'])
def process_noun():
    # Fetching news
    results = []
    sentiment_counts = [0, 0, 0]
    overall_polarity_score = 0

    search_word = request.form['noun']
    no_of_articles = 10

    url = f'https://newsapi.org/v2/everything?q={search_word}&sortBy=relevancyAt&searchIn=title&pageSize={no_of_articles}&apiKey=5081961eb5d94cabaf80779f9ecbf515&language=en'

    response = requests.get(url)
    data = response.json()
    articles = data.get('articles')

    sentiment_counts=[0,0,0]

    for article in articles:
        title = article['title']
        url = article['url']
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        authors = soup.find_all('a', {'rel': 'author'})

        text = ""
        for p in soup.find_all('p'):
            text += p.text

        # Summarize the text
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer = LexRankSummarizer()
        summary = summarizer(parser.document, sentences_count=3)
        summarized_text = ' '.join(str(sentence) for sentence in summary)

        # Analyze sentiment using VADER
        analyzer = SentimentIntensityAnalyzer()
        sentiment_scores = analyzer.polarity_scores(summarized_text)

        sentiment = ''
        if sentiment_scores['compound'] >= 0.05:
            sentiment = 'Positive'
        elif sentiment_scores['compound'] <= -0.05:
            sentiment = 'Negative'
        else:
            sentiment = 'Neutral'

        # Update sentiment counts and overall polarity score
        if sentiment == "Positive":
            sentiment_counts[0] += 1
        elif sentiment == "Negative":
            sentiment_counts[1] += 1
        else:
            sentiment_counts[2] += 1

        overall_polarity_score += sentiment_scores['compound']

        # Store the results
        result = {
            "title": title,
            "url": url,
            "sentiment": sentiment,
            "positive_count": sentiment_counts[0],
            "negative_count": sentiment_counts[1],
            "neutral_count": sentiment_counts[2],
            "overall_polarity_score": sentiment_scores['compound']
        }
        results.append(result)

    # Calculate average sentiment counts and overall polarity score
    average_sentiment_counts = [count / len(articles) for count in sentiment_counts]
    average_polarity_score = overall_polarity_score / len(articles)


    # Calculate average sentiment counts and overall polarity score
    average_sentiment_counts = [count / len(articles) for count in sentiment_counts]
    average_polarity_score = overall_polarity_score / len(articles)

    sentiments = ["Positive", "Negative", "Neutral"]
    colors = ['#00ff00', '#ff0000', '#ffff00']

    plt.bar(sentiments, average_sentiment_counts, color=colors)
    plt.title("Average Sentiment Analysis")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.show()

    # Create the pie chart using average sentiment counts
    labels = ['Positive', 'Negative', 'Neutral']
    sizes = average_sentiment_counts
    colors = ['#00ff00', '#ff0000', '#ffff00']
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.title('Average Sentiment Analysis Results')
    plt.show()

    return json.dumps(results)

if __name__ == '__main__':
    app.run(debug=True)


    