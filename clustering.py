# pip install nltk
# pip install wordcloud
# pip install Sastrawi
# pip install stop-words

import nltk
import numpy as np
import pandas as pd
import re
import pylab as pl
import matplotlib.pyplot as plt
import json

from nltk.tokenize import WordPunctTokenizer
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn import metrics
from mpl_toolkits.mplot3d import Axes3D

from matplotlib import pyplot

# Specify the path to your CSV file
file_path = 'sosis_keju_indomaret.csv'

# Read CSV into a pandas DataFrame
df = pd.read_csv(file_path, delimiter=";")

def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)

    return input_txt

# remove Twitter usernames (mentions)
df['clean_text'] = np.vectorize(remove_pattern)(df['full_text'], "@[\w]*")

# Remove special characters, numbers, punctuations, and links
df['clean_text'] = df['clean_text'].apply(lambda x: ' '.join(re.sub("(\w+:\/\/\S+)|[^a-zA-Z#]", " ", x).split()))

# split text into individual words or tokens
tokenized_tweet = df['clean_text'].apply(lambda x: x.split())

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Initialize the Sastrawi stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Apply stemming and convert to space-separated strings
tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x])

# remove words with a length less than to 3 characters
tokenized_tweet = tokenized_tweet.apply(lambda x: [w for w in x if len(w)>3])

for i in range(len(tokenized_tweet)):
    tokenized_tweet[i] = ' '.join(tokenized_tweet[i])

df['clean_text'] = tokenized_tweet

from sklearn.feature_extraction.text import TfidfVectorizer
from stop_words import get_stop_words

# Get the list of Indonesian stop words
indonesian_stop_words = [
    'indomaret', 'sosis', 'keju', 'adalah', 'dari', 'dalam', 'yang', 'oleh', 'pada', 'untuk',
    'dengan', 'sebagai', 'atau', 'saya', 'kami', 'anda', 'mereka', 'kita',
    'belum', 'telah', 'akan', 'jika', 'sudah', 'sekarang', 'bukan',
    'lagi', 'saat', 'karena', 'sehingga', 'itulah', 'bagi', 'maka', 'apakah',
    'tersebut', 'tentang', 'melakukan', 'sebelum', 'sesuatu', 'setelah', 'dapat',
    'harus', 'masih', 'mungkin', 'lebih', 'terlalu', 'kepada', 'lain', 'sama',
    'masa', 'doang', 'bisa', 'udah', 'juga', 'pernah', 'buat', 'tapi', 'jadi',
    'banget', 'kalau', 'punya', 'cuma', 'kalo', 'nanti', 'terus', 'malah', 'mana', 'wkwk',
    'huft', 'kenapa', 'makan'
]

# Create the TfidfVectorizer with Indonesian stop words and use_idf parameter
tfidf_vect = TfidfVectorizer(
    analyzer='word',
    ngram_range=(1, 1),
    stop_words=indonesian_stop_words,
    min_df=0.0001,
    use_idf=True
)

# Fit and transform the text data
tfidf_matrix = tfidf_vect.fit_transform(df['clean_text'])

from sklearn.cluster import KMeans
# implement kmeans
num_clusters = 5
km = KMeans(n_clusters=num_clusters)
km.fit(tfidf_matrix)
clusters = km.labels_.tolist()

# create DataFrame films from all of the input files.
tweets = {'Tweet': df["clean_text"].tolist(), 'Cluster': clusters}
clustered_tweet = pd.DataFrame(tweets, index = [clusters])

cluster_word_counts = {}
for cluster_id in range(num_clusters):
    cluster_df = clustered_tweet[clustered_tweet['Cluster'] == cluster_id]
    cluster_text = ' '.join(cluster_df['Tweet'])
    words = cluster_text.split()

    # Filter out stop words
    filtered_words = [word for word in words if word not in indonesian_stop_words]

    word_counts = {}
    for word in filtered_words:
        if word in word_counts:
            word_counts[word] += 1
        else:
            word_counts[word] = 1
    cluster_word_counts[cluster_id] = word_counts

cluster_word_counts_json = json.dumps(cluster_word_counts)

data_cluster = []

for cluster_id, word_counts in cluster_word_counts.items():
    sorted_word_counts = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    top_words = [word for word, count in sorted_word_counts[:10]]
    data_cluster.append({'Cluster': cluster_id, 'Top Words': ', '.join(top_words)})

data_cluster = pd.DataFrame(data_cluster)

from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Create a WordCloud object for each cluster and generate word clouds
num_clusters = len(cluster_word_counts)
rows = (num_clusters + 1) // 2  # Adjust the number of rows as needed
fig, axes = plt.subplots(rows, 2, figsize=(12, 8))

for cluster_id, word_counts in cluster_word_counts.items():
    sorted_word_counts = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    top_words = dict(sorted_word_counts[:10])
    wordcloud = WordCloud(width=400, height=200, background_color='white').generate_from_frequencies(top_words)

    row = cluster_id // 2
    col = cluster_id % 2

    axes[row, col].imshow(wordcloud, interpolation='bilinear')
    axes[row, col].set_title(f"Cluster {cluster_id+1} Word Cloud")
    axes[row, col].axis('off')

# If there are an odd number of clusters, hide the last subplot
if num_clusters % 2 != 0:
    fig.delaxes(axes[rows - 1, 1])

plt.tight_layout()

plt.savefig('static/word-clouds.png', format='png')