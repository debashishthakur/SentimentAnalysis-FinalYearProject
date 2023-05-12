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
        <html>
            <head>
                <link href="https://fonts.googleapis.com/css?family=Roboto:400,700&display=swap" rel="stylesheet">
                <style>
                    body {
                        background: linear-gradient(135deg, #292929, #121212);
                        display: flex;
                        flex-direction: column;
                        justify-content: center;
                        align-items: center;
                        height: 100vh;
                        font-family: Arial, sans-serif;
                        color: #ffffff;
                        font-family: 'Roboto', Arial, sans-serif;
                    }

                    h1 {
                        color: #ffffff;
                        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
                        margin-bottom: 80px;
                    }

                    .container {
                        display: flex;
                        justify-content: center;
                        align-items: center;
                    }

                    .box {
                        width: 400;
                        height: 200;
                        border-radius: 10px;
                        text-align: center;
                        line-height: 200px;
                        font-size: 18px;
                        margin: 0 43px;
                        transition: box-shadow 0.3s;
                    }

                    .box-twitter {
                        background: linear-gradient(45deg, #00c6fb, #005bea);
                        box-shadow: 0 0 10px rgba(17, 173, 242, 0.4);
                        color: #ffffff;
                    }

                    .box-news {
                        background: linear-gradient(45deg, #fc4a1a, #f7b733);
                        box-shadow: 0 0 10px rgba(236, 119, 50, 0.4);
                        color: #ffffff;
                    }

                    .box-review {
                        background: linear-gradient(45deg, #7f00ff, #e100ff);
                        box-shadow: 0 0 10px rgba(181, 78, 255, 0.4);
                        color: #ffffff;
                    }

                    .box:hover {
                        box-shadow: 0 0 20px rgba(163, 177, 198, 0.8);
                    }
                    a{
                        text-decoration: none;
                    }
                </style>
            </head>
            <body>
                <h1>Web-Sentiment Analysis</h1>
                <div class="container">
                    <a class="box box-twitter" href="/search">Twitter Review</a>
                    <a class="box box-news" href="/search">News Review</a>
                    <a class="box box-review" href="/search">Review Analysis</a>
                </div>
            </body>
        </html>
    '''

@app.route('/search')
def search():
    return '''
        <html>
            <head>
                <link href="https://fonts.googleapis.com/css?family=Roboto:400,700&display=swap" rel="stylesheet">
                <style>
                    body {
                        background: linear-gradient(135deg, #292929, #121212);
                        display: flex;
                        justify-content: center;
                        align-items: center;
                        height: 100vh;
                        font-family: 'Roboto', Arial, sans-serif;
                        color: #ffffff;
                    }

                    .form-container {
                        background: linear-gradient(45deg, #00c6fb, #005bea);
                        box-shadow: 0 0 10px rgba(17, 173, 242, 0.4);
                        padding: 20px;
                        border-radius: 20px;
                        text-align: center;
                    }

                    .form-label {
                        font-weight: bold;
                        margin-bottom: 10px;
                        color: #ffffff;
                    }

                    .form-input {
                        padding: 8px;
                        border: none;
                        border-radius: 6px;
                        width: 200px;
                        font-size: 16px;
                    }

                    .form-button {
                        padding: 10px 20px;
                        border: none;
                        border-radius: 6px;
                        background-color: #4b7bec;
                        color: #ffffff;
                        cursor: pointer;
                        font-size: 16px;
                        margin-top: 10px;
                    }

                    .form-button:hover {
                        background-color: #3867d6;
                    }
                </style>
            </head>
            <body>
                <div class="form-container">
                    <form action="/process-noun" method="post">
                        <label class="form-label" for="noun">Enter a noun:</label>
                        <input class="form-input" type="text" id="noun" name="noun" required>
                        <button class="form-button" type="submit">Submit</button>
                    </form>
                </div>
            </body>
        </html>
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

    html_response = '''
        <html>
            <head>
                <link href="https://fonts.googleapis.com/css?family=Roboto:400,700&display=swap" rel="stylesheet">
                <style>
                    body {
                        background: linear-gradient(135deg, #292929, #121212);
                        font-family: 'Roboto', Arial, sans-serif;
                        color: #ffffff;
                        padding: 30px;
                    }

                    .result-container {
                        background: linear-gradient(135deg, #292929, #121212);
                        box-shadow: 0 0 10px rgba(17, 173, 242, 0.4);
                        padding: 20px;
                        border-radius: 20px;
                        margin-bottom: 20px;
                    }

                    .result-title {
                        font-size: 24px;
                        font-weight: bold;
                        margin-bottom: 10px;
                        color: #ffffff;
                    }

                    .result-sentiment {
                        font-size: 18px;
                        color: #ffffff;
                    }

                    .result-polarity {
                        font-size: 16px;
                        margin-bottom: 10px;
                        color: #ffffff;
                    }

                    .result-link {
                        font-size: 16px;
                        color: #ffffff;
                        text-decoration: none;
                    }

                    .result-link:hover {
                        text-decoration: underline;
                    }

                    .result-chart {
                        margin-top: 20px;
                    }

                    h1 {
                        color: #ffffff;
                        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
                        margin-bottom: 30px;
                        text-align: center;
                    }

                </style>
            </head>
            <body>
                <h1>News Results</h1>
    '''

    for result in results:
        html_response += '''
            <div class="result-container">
                <h2 class="result-title">{}</h2>
                <p class="result-sentiment">Sentiment: {}</p>
                <p class="result-polarity">Overall Polarity Score: {}</p>
                <p class="result-link"><a href="{}" target="_blank">Read More</a></p>
                <hr/>
            </div>
        '''.format(result['title'], result['sentiment'], result['overall_polarity_score'], result['url'])


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
    html_response += '''
        <div class="result-container">
            <p class="result-chart-title"><strong>Sentiment Analysis Chart:</strong></p>
            <img class="result-chart" src='data:image/png;base64,{}' alt='Sentiment Analysis Chart'>
            <hr/>
        </div>
    '''.format(encoded_string)

    
    return html_response

if __name__ == '__main__':
    app.run(debug=True)
