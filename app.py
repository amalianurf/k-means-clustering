from flask import Flask, render_template
from clustering import clustered_tweet, data_cluster, cluster_word_counts_json

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', data_cluster=data_cluster, cluster_word_counts=cluster_word_counts_json)

@app.route('/tweet-data')
def tweet_data():
    return render_template('tweet-data.html', clustered_tweet=clustered_tweet)

if __name__ == '__main__':
    app.run(debug=True)