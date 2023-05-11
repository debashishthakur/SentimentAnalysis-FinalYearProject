
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
    result={}
    search_word = request.form['noun']
    no_of_articles = 10

    url = f'https://newsapi.org/v2/everything?q={search_word}&sortBy=relevancyAt&pageSize={no_of_articles}&apiKey=5081961eb5d94cabaf80779f9ecbf515&language=en'


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

        # Create a bar chart to display the sentiment analysis results
        sentiments = ["Positive", "Negative", "Neutral"]
        counts = [0, 0, 0]
        if sentiment == "Positive":
            counts[0] = 1
        elif sentiment == "Negative":
            counts[1] = 1
        else:
            counts[2] = 1

        plt.bar(sentiments, counts)
        plt.title("Overall Sentiment Analysis")
        plt.xlabel("Sentiment")
        plt.ylabel("Count")
        plt.show()


        # Analyze each sentence and keep track of the sentiment scores
        pos_count = 0
        neg_count = 0
        neu_count = 0
        pol_score = 0
        for sentence in summary:
            score = analyzer.polarity_scores(str(sentence))
            if score['compound'] > 0.1:
                pos_count += 1
            elif score['compound'] < -0.1:
                neg_count += 1
            else:
                neu_count += 1 
        pol_score = pol_score+ score['compound']
                
        # Create a pie chart to display the sentiment analysis results
        labels = ['Positive', 'Negative', 'Neutral']
        sizes = [pos_count, neg_count, neu_count]
        colors = ['#00ff00', '#ff0000', '#ffff00']
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        plt.axis('equal')
        plt.title('Sentiment Analysis Results')
        plt.show()

        # Store the results
        result = {
            "title": title,
            "url": url,
            "sentiment": sentiment,
            "positive_count": counts[0],
            "negative_count": counts[1],
            "neutral_count": counts[2],
            "overall_polarity_score": sentiment_scores['compound']
        }
        results.append(result)

        # print(response.json())
    return json.dumps(results)

if __name__ == '__main__':
    app.run(debug=True)


    