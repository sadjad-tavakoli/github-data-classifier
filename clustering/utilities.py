import os

from numpy.dual import norm
from numpy.ma import dot
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


def tf_idf_calculator():
    whole_data = []
    documents_path = os.getcwd() + "\clustering\issues_comments"
    listdir = os.listdir(documents_path)
    files_count = len(listdir)
    for index, file_name in enumerate(listdir):
        print("== reading file {} of {} ".format(index, files_count))
        file = open(documents_path + "\\" + file_name, encoding="utf-8")
        content = file.read()
        whole_data.append(content)
        file.close()

    tfidf_vectorizer = TfidfVectorizer(sublinear_tf=True)
    tf_idf_vectors = tfidf_vectorizer.fit_transform(whole_data)
    return tf_idf_vectors


def cos_similarity(vec1, vec2):
    return dot(vec2, vec1) / (norm(vec2) * norm(vec1))


def inertia_calculator(clusters):
    total_sum_of_squares = 0
    for index, documents in enumerate(clusters.values()):

        average = sum([item[1] for item in documents]) / len(documents)
        for item in documents:
            total_sum_of_squares += (1 - cos_similarity(item[1], average)) ** 2

    return total_sum_of_squares


def clusters_dictionary(data, labels):
    result = {}
    documents_path = os.getcwd() + "\clustering\issues_comments"
    listdir = os.listdir(documents_path)

    for index, item in enumerate(data):
        values = result.get(labels[index], [])
        values.append((listdir[index], item))
        result.update({labels[index]: values})
    return result


def get_top_n_words(words_list, n, limited_words):
    vec = CountVectorizer().fit(words_list)
    bag_of_words = vec.transform(words_list)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    purified = list(set([item[0] for item in words_freq]) - set(limited_words))
    return purified[:n]
