from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import classification_report
import json
# from pyspark.ml.clustering import KMeans
from pyspark.sql import SparkSession
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from pyspark.ml.feature import IDF


spark = SparkSession.builder.appName('JSONRead').getOrCreate()
test_data = spark.read.json('data/Movie_reviews.json')

test_data.createOrReplaceTempView("Test_Data")
testing_text = spark.sql("SELECT reviewText from Test_Data")

testing_text.show()
# testing_class.show()

testing_text = []
testing_class = []

with open('data/training.json') as outfile:
    data = json.load(outfile)

    for item in data:
        testing_text.append(item["reviewText"])
        testing_class.append(item["classification"])

# test_data = spark.read.json('data/training.json')
#
# test_data.createOrReplaceTempView("Test_Data")
# testing_text = spark.sql("SELECT reviewText from Test_Data")

# with open('data/Movie_reviews.json') as testf:
#     test_data = json.load(testf)
#     for item in test_data:
#         testing_text.append(item["reviewText"])

training_text = [
    "This movie was very bad. It wasted my time.",
    "Movie was short and disappointing and bad",
    "Quick film that left me happy. Was a great and amazing film",
    "I didn't like the movie at all. The plot was bad and horrible",
    "This movie was everything that I expected. Made the family really happy and was perfect and wonderful",
    "Movie was very good in this twist on the classic story",
    "The casting is excellent and the music and theme was good",
    "I immediately fell in love with it because it was very good and one of my favorite movies",
    "I totally liked the movie and the songs that the they did. An excellent and amazing ending!",
    "Nothing mad sense and it was a bad movie",
    "This was a garbage of a film and it was very bad. I wouldn't waste money on it.",
    "I wasted my money on this horrible and bad movie. I hated the ending and the plot mad no sense. Bad movie."
    "This was a lousy film with horrible acting. The girls had no talent and made the film bad.",
    "I wouldn't spend time watching an annoying, unintelligent, no talent, horrible, bad film."
    "Movie was talentless and pointless which made it a horrible movie",
    "The actors in the movie was stupid and dullwitted which flat out wasted my time. This was a bad movie.",
    "Nothing more disappointing this this movie. Sad plot and bad ending.",
    "Bad movie, horrible ending, and unacceptable price.",
    "This was an outstanding and amazing movie. Hands down the best movie I have ever seen. I loved everything about it. Was a wonderful movie.",
    "Movie is very good in this twist on the classic story. Satisified with my movie of choice. Great and fantastic movie.",
    "This has been a favorite movie of mine for a long time. I would watch it again since I had such a great and enjoyable time.",
    "Enjoyed this movie. The character does a good job. I would watch it again. Loved the movie.",
    "I liked this movie. It was a wonderful and one of my favorites. Great film!",
    "Immediately fell in love with this movie. Great actors and an amazing plot.",
    "I rate this a zero star. It was a horrible movie.",
    "I rate this a one start. Horrible movie and was a waste of my time.",
    "This is a wonderful movie and a must see!",
    "A great film to see! A good movie. A great purchase.",
    "I failed to see the point of this movie. There was bad action, horrible plot.",
    "This movie was pretty pointless. Bad acting and horrible plot.",
    "One of the best movies out there. One of the best movies I have watched. I loved it. A favorite!"
]

vectorizer = TfidfVectorizer(stop_words='english',)
X = vectorizer.fit_transform(training_text)
true_k = 2
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
model.fit(X)

order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(true_k):
    print("Cluster %d:" % i),
    for ind in order_centroids[i, :15]:
        print(' %s' % terms[ind]),
    print


Y = vectorizer.transform(testing_text)
prediction = model.predict(Y)

print(Y)


feature_names = vectorizer.get_feature_names()
corpus_index = [n for n in testing_text]
import pandas as pd

frame = pd.DataFrame(X.toarray(),columns=["acting",	"action",	"actors",	"amazing",	"annoying",	"bad",	"best",	"casting",	"character",	"choice",	"classic",	"did",	"didn",	"disappointing",	"does",	"dullwitted",	"ending",	"enjoyable",	"enjoyed",	"excellent",	"expected",	"failed",	"family",	"fantastic",	"favorite",	"favorites",	"fell",	"film",	"flat",	"garbage",	"girls",	"good",	"great",	"hands",	"happy",	"hated",	"horrible",	"immediately",	"job",	"left",	"like",	"liked",	"long",	"lousy",	"love",	"loved",	"mad",	"money",	"movie",	"movies",	"music",	"outstanding",	"perfect",	"plot",	"point",	"pointless",	"pretty",	"price",	"purchase",	"quick",	"rate",	"really",	"sad",	"satisified",	"seen",	"sense",	"short",	"songs",	"spend",	"star",	"start",	"story",	"stupid",	"talent",	"talentless",	"theme",	"time",	"totally",	"twist",	"unacceptable",	"unintelligent",	"waste",	"wasted",	"watch",	"watched",	"watching",	"wonderful",	"wouldn",	"zero"])
frame.to_csv("data/tfidf.csv", sep='\t', encoding='utf-8',index=False)
print(frame)

ax = frame.plot(kind='scatter', x='good', y='bad', alpha=0.1, s=200)
ax.set_xlim(-0.05, 0.5)
ax.set_ylim(-0.05, 0.5)

plt.show(block=True)
#===========================================

# kmeans = KMeans().setK(2).setSeed(1)
# model = kmeans.fit(testing_text)
# predictions = model.transform(testing_text)
# centers = model.clusterCenters()

word_prediction = []
for x in range(len(prediction)):
    if prediction[x] == 1:
        word_prediction.append("satisfied")
    if prediction[x] == 0:
        word_prediction.append("unsatisfied")

print(prediction)

# print(classification_report(testing_class, word_prediction, target_names=["unsatisfied", "satisfied"]))


