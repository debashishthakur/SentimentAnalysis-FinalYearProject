import requests
import json
from flask import Flask, request
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from bs4 import BeautifulSoup
import nltk
import base64
import re

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

    search_word = request.form['noun']
    no_of_articles = 10

    url = f'https://newsapi.org/v2/everything?q={search_word}&sortBy=relevancyAt&searchIn=title&pageSize={no_of_articles}&apiKey=5081961eb5d94cabaf80779f9ecbf515&language=en'

    response = requests.get(url)
    data = response.json()
    articles = data.get('articles')

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

        # Store the results
        result = {
            "title": title,
            "url": url,
            "sentiment": sentiment,
            "overall_polarity_score": sentiment_scores['compound']
        }
        results.append(result)

    html_response = "<h1>News Results</h1>"

    for result in results:
        html_response += f"<h2>{result['title']}</h2>"
        html_response += f"<p><strong>Sentiment:</strong> {result['sentiment']}</p>"
        html_response += f"<p><strong>Overall Polarity Score:</strong> {result['overall_polarity_score']}</p>"
        html_response += f"<p><a href='{result['url']}' target='_blank'>Read More</a></p>"
        html_response += "<hr/>"

        # Analyze each article separately and create the pie chart for sentiment analysis
        summarized_text = summarizer(parser.document, sentences_count=3)
        sentiment_counts = [0, 0, 0]
        for sentence in summarized_text:
            sentiment_scores = analyzer.polarity_scores(str(sentence))
            if sentiment_scores['compound'] >= 0.05:
                sentiment_counts[0] += 1
            elif sentiment_scores['compound'] <= -0.05:
                sentiment_counts[1] += 1
            else:
                sentiment_counts[2] += 1

    # Create the pie chart using sentiment counts
    labels = ['Positive', 'Negative', 'Neutral']
    sizes = sentiment_counts
    colors = ['#00ff00', '#ff0000', '#ffff00']
    plt.figure()
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')

    # Sanitize the title to remove invalid characters from the filename
    sanitized_title = re.sub(r'[^\w\s-]', '', result['title']).strip()
    sanitized_title = re.sub(r'[-\s]+', '-', sanitized_title)

    # Save the chart to a file
    chart_filename = f'chart_{sanitized_title}.png'
    plt.savefig(chart_filename)
    plt.close()

    # Encode the chart image as a base64 string
    with open(chart_filename, 'rb') as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

    # Add the chart to the HTML response
    html_response += "<p><strong>Sentiment Analysis Chart:</strong></p>"
    html_response += f"<img src='data:image/png;base64,{encoded_string}' alt='Sentiment Analysis Chart'>"
    html_response += "<hr/>"
    
    return html_response

if __name__ == '__main__':
    app.run(debug=True)
