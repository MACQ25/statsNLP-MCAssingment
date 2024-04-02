## The corpus (would normally load from a file or database)
docs = [
    "I have Magic Tavern pins!",
    "Magic Tavern pins with Wereboar design.",
    "I am a Wereboar",
    "This wereboar let me into this tavern"
]
## create and fit vectorizer
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()

vectors = vectorizer.fit_transform(docs)

## vectorize a new document based on the same vocabulary
new_vector = vectorizer.transform(['I love "Hello from the Magic Tavern" pins. I am not a Wereboar.'])

## get similarities between new document and the originals
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
cosine_sim = cosine_similarity(new_vector, vectors)
euclidean_sim = euclidean_distances(new_vector, vectors)
print(cosine_sim, "\n", euclidean_sim)