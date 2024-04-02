from csv import DictReader
from random import randint, shuffle
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances


NUM_RECS = 101  # number of recommendations to return to the user


def load_articles(filename, num=None, filetype="csv"):
    """Returns a list of articles loaded from a json or csv file with a header.
    Each article is a dictionary. If you use one of the files provided, each
    article will have a "title" and "text" field.

    has been modified to try and prevent two of the same article from being
    loaded in (checks if the body of the article is exactly the same)

    filename = name of file to load
    num = number of articles to load (random sample), or None for all articles
    filtype = "csv" or "json" """
    articles = []
    if filetype == "csv":
        with open(filename, encoding="utf-8") as csvfile:
            reader = DictReader(csvfile)
            for row in reader:
                if not any(article["text"] == row["text"] for article in articles):
                    articles.append(row)
    elif filetype == "json":
        with open(filename, encoding="utf-8") as jsonfile:
            articles = json.loads(jsonfile.read())
    for row in articles:
        if row["title"] is None:
            row["title"] = row["text"][:30]
    if num:
        shuffle(articles)
        articles = articles[:num]
    print(len(articles), "articles loaded")
    return articles


def init_recommendations(n, articles):
    """This generates n random recommendations."""
    recommendations = []
    for _ in range(n):
        article = randint(0, len(articles) - 1)
        while article in recommendations:
            article = randint(0, len(articles) - 1)
        recommendations.append(article)
    return recommendations


def display_recommendations(recommendations, articles):
    """Displays recommendations. The recommendations parameter should be a list
    of index numbers representing the recommended articles."""
    print("\n\n\nHere are some new recommendations for you:\n")
    for i in range(len(recommendations)):
        art_num = recommendations[i]
        if i == 4*(len(recommendations) // 5):
            print("\nAnd some other news:\n")
        print(str(i + 1) + ".", articles[art_num]["title"])


def display_article(art_num, articles):
    """Displays article 'art_num' from the articles"""
    print("\n\n")
    print("article", art_num)
    print("=========================================")
    print(articles[art_num]["title"])
    print()
    print(articles[art_num]["text"])
    print("=========================================")
    print("\n\n")


# Information and use of the SKLearn library and methods was acquired from the materials found
# within the course shell in myCanvas
def make_vectors(items: list):
    """
    Function used to instantiate a CountVectorizer object from the SKlearn library
    and establish its vocabulary as that of the collective contents of the articles
    list (the parameter), the CountVectorizer has been set up to look for a custom
    token_pattern, to ignore words that appear in above 30% of articles and also
    those that only appear in 2 or less
    :param items: List of all articles to be vectorized
    :return: Vectorizer object and list of vectorized articles
    """
    vect = CountVectorizer(max_df=0.3, min_df=2, token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]\w*\b|\b\d\d\d\d\b")
    doc = []
    for art in items:
        if art.__contains__("text"):
            doc.append(art["text"])
    docVects = vect.fit_transform(doc)
    return vect, docVects


# Sorted built-in function was consulted from:
# https://www.simplilearn.com/tutorials/python-tutorial/sort-in-python
# Zip built-in function was consulted from:
# https://www.geeksforgeeks.org/python-convert-two-lists-into-a-dictionary/
def get_cosine_similarity(article: str, vectorizer, vectors):
    """
    Function used generate the cosine similarity of the current
    article with the rest of the vectorized articles, it takes in
    the article and then calls the cosine_similarity function
    passing in said article and all given vectors

    It then creates an ordered and indexed dictionary that is
    ordered from "Highest similarity to lowest" which is then
    returned
    :param article: String being compared with the rest of vectors
    :param vectorizer: Object containing all the vocabulary, meant to be instantiated by make_vectors
    :param vectors: All vectors from every available article
    :return: Reverse sorted dictionary of cosine_similarity results, with the indexes of each vectors as the keys and float as their values
    """
    csm = cosine_similarity(vectorizer.transform([article]), vectors)
    BOC = dict(zip(range(0, len(csm[0])), csm[0]))
    return dict(sorted(BOC.items(), key=lambda x:x[1], reverse=True))


# built-in any() function was consulted from here:
# https://www.pythonmorsels.com/any-and-all/
def no_repeat_title(articles, toRecommend: list, index: int):
    """
    Boolean evaluation function meant to search the contents of articles list as referenced
    by the indexes contained in "toRecommned", if one of the indexes of "toRecommend" shares
    a title already with the current index, it returns false and said index is ignored,
    otherwise it returns true and said article will be addedto the list of articles recommended
    to the user
    :param articles: All available articles in a list
    :param toRecommend: List of indexes that reference articles in the articles list
    :param index: Current index being evaluated
    :return: Boolean evaluation determining if the value has a duplicate
    """
    return not any(articles[article]["title"] == articles[index]["title"] for article in toRecommend)


def add_recommendations(lowRange, topRange, indexList, articles, recommendations, step=1):
    """
    Function used to create a set number of article recommendations using
    a range, lowRange being the starting index and topRange the last index,
    in the case that one enters the parameter step as negative, you can
    reverse them.

    The indexList is an ordered list of indexes referencing a
    dictionary of cosine-similarity results

    The articles, which contains all relevant articles, their body, id and title,
    is entered and passed in to the no_repeat_tittle() function, to make sure
    no two titles are the same

    Recommendations is the list into which a value from the index list is
    appended after the boolean operations and the offset has been added (if needed)

    :param lowRange: The Lowest index from which to start in the index list
    :param topRange: The Highest index to reach in the index list
    :param indexList:
    :param articles:
    :param recommendations:
    :param step: How much to add per iteration of the loop to the current value, make sure to flip the values of lowRange and topRange in the case this is set up as negative
    :return:
    """
    offset = 0
    for rel in range(lowRange, topRange, step):
        noRep = False
        while not noRep:
            simIndex = indexList[0 + (rel + offset)]
            if no_repeat_title(articles, recommendations, simIndex):
                recommendations.append(simIndex)
                noRep = True
            else:
                offset = offset + 1 if step > 0 else offset - 1


def new_recommendations(last_choice, n, articles, vectorizer, vectors):
    """
    Function used to give the user new recommendations based on their last choice
    it makes use of the get_cosine_similarity function, along with the vectorizer
    object and vectors list in order to create a list of floats and indexes to use,
    compare and reference other articles and give back the user entries that are
    relevant to their last selection, this will also add articles on the opposite
    end of the spectrum, recommending articles that are an almost complete mismatch
    with the one selected by the user

    A 5th of all recommendations, all allocated at the end of the list, will always
    be completely unrelated to the users selection
    :param last_choice: Index of the last article chosen by the player
    :param n: Number of recommendations to make
    :param articles: List of all articles available
    :param vectorizer: Vectorizer object with the vocabulary of all documents
    :param vectors: All vectors from each article
    :return: A list of ints that are indexes referencing articles, these are the recommendations
    """

    cosSims = get_cosine_similarity(articles[last_choice]["text"], vectorizer, vectors)
    indexList = list(cosSims.keys())
    recommendations = []

    currentRange = 4*(n//5)+1

    add_recommendations(1, currentRange, indexList, articles, recommendations)

    currentRange = (n//5)+1

    add_recommendations(len(indexList)-1, (len(indexList) - currentRange - 1), indexList, articles, recommendations, -1)

    return recommendations


def main():
    articles = load_articles('data\\bbc_news.csv', filetype="csv")
    vectorizer, bokVectors = make_vectors(articles)
    print("\n\n")
    recs = init_recommendations(NUM_RECS, articles)
    while True:
        display_recommendations(recs, articles)
        choice = int(input("\nYour choice? ")) - 1
        if choice < 0 or choice >= len(recs):
            print("Invalid Choice. Goodbye!")
            break
        display_article(recs[choice], articles)
        input("Press Enter")
        recs = new_recommendations(recs[choice], NUM_RECS, articles, vectorizer, bokVectors)


if __name__ == "__main__":
    main()
