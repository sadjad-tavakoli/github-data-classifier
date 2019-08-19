import ast
import os

import gensim
import matplotlib.pyplot as plt
from gensim import models
from gensim.models import CoherenceModel


def get_processed_data(tf_idf=False):
    whole_data = []
    documents_path = os.getcwd() + "\\topic_modeling\issues_words"
    listdir = os.listdir(documents_path)
    files_count = len(listdir)
    for index, file_name in enumerate(listdir):
        print("== reading file {} of {} ".format(index, files_count))
        file = open(documents_path + "\\" + file_name, encoding="utf-8")
        content = file.read()
        words_list = ast.literal_eval(content)
        whole_data.append(words_list)
        file.close()
    dictionary = gensim.corpora.Dictionary(whole_data)
    dictionary.filter_extremes()
    corpus = [dictionary.doc2bow(doc) for doc in whole_data]

    if tf_idf:
        tfidf = models.TfidfModel(corpus)
        corpus = tfidf[corpus]

    return whole_data, dictionary, corpus


def find_model_number():
    processed_docs, dictionary, corpus = get_processed_data()

    coherence_ldas = []
    for number in range(1, 20, 1):
        print("== on {} models running ===".format(number))
        lda_model = gensim.models.LdaMulticore(corpus, num_topics=number, id2word=dictionary,
                                               chunksize=100, workers=None)
        coherence_model_lda = CoherenceModel(model=lda_model, texts=processed_docs,
                                             dictionary=dictionary)
        coherence_ldas.append(coherence_model_lda.get_coherence())

    x = range(1, 20, 2)
    plt.plot(x, coherence_ldas)
    plt.xlabel("Num Topics")
    plt.ylabel("Coherence score")
    plt.legend("coherence_values", loc='best')
    plt.show()


def topic_modeling(topic_number, tf_idf=False):
    processed_docs, dictionary, bow_corpus = get_processed_data(tf_idf)

    lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=topic_number,
                                           id2word=dictionary, workers=None, chunksize=100)

    result_report(dictionary, lda_model, processed_docs)


def mallet_topic_modeling(topic_number, tf_idf=False):
    processed_docs, dictionary, corpus = get_processed_data(tf_idf)

    os.environ['MALLET_HOME'] = 'C:\mallet-2.0.8'
    mallet_path = "C:\\mallet-2.0.8\\bin\\mallet"

    lda_model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus,
                                                 num_topics=topic_number, id2word=dictionary)
    result_report(dictionary, lda_model, processed_docs)


def result_report(dictionary, lda_model, processed_docs):
    coherence_model_lda = CoherenceModel(model=lda_model, texts=processed_docs,
                                         dictionary=dictionary, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()

    print('\nCoherence Score: ', coherence_lda)
    print("\n top topics using Bag of Words:  in {} files\n".format(len(processed_docs)))

    result_file = open("reports\\topic_modeling_result.txt", 'w', encoding="utf-8")
    result_file.write('\nCoherence Score: {}\n'.format(coherence_lda))
    result_file.write(
        '\n\n \t top topics using Bag of Words:  in {} files\n\n'.format(len(processed_docs)))

    for idx, topic in lda_model.print_topics(-1):
        print('Topic: {} \nWords: {}'.format(idx, topic))
        result_file.write(
            '------------  Topic: {} ------------\n\nWords: {}\n\n'.format(idx, topic))
    result_file.close()
