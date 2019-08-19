import os

import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

from clustering.utilities import tf_idf_calculator, inertia_calculator, clusters_dictionary, \
    get_top_n_words
from tokenizer import text_processing


def agglomerative_elbow():
    tss = {}
    for k in range(1, 20):
        tss[k] = inertia_calculator(calculate_clusters(k))
        print("inertia for {} is {}".format(k, tss[k]))

    plt.figure()
    plt.plot(list(tss.keys()), list(tss.values()))
    plt.xlabel("Number of cluster")
    plt.ylabel("TSS")
    plt.show()
    plt.savefig('foo.png')


def agglomerative_clustering(n_clusters):
    n_top = n_clusters if n_clusters < 10 else 10
    clusters = calculate_clusters(n_clusters)
    sorted_clusters = sorted(clusters.items(), key=lambda kv: len(kv[1]), reverse=True)
    result = {}

    print("==== result processing ====")
    for index, cluster in enumerate(sorted_clusters[0:n_top]):
        cluster_words = []
        documents = cluster[1]
        for pair in documents:
            file_name = pair[0]
            file = \
                open(os.getcwd() + "\clustering\issues_comments\\" + file_name, encoding="utf-8")
            content = file.read()
            cluster_words += text_processing(content)
            file.close()

        result.update({index: get_top_n_words(cluster_words, 20, [])})

        print("=== cluster {} words processing ====".format(index))
        print("=== cluster {} done ====".format(index))

    result_file = open("reports\\agglomerative clustering result in {} clusters.txt"
                       .format(n_clusters), 'w', encoding="utf-8")

    for idx, topic in result.items():
        print('------------  cluster {} ------------\n\nTop Words: {}\n\n'.format(idx, topic))
        result_file.write(
            '------------  cluster {} ------------\n\nTop Words: {}\n\n'.format(idx, topic))
    result_file.close()

    print("\n *** you can find the result in a .txt file in the project's root directory. ***")


def calculate_clusters(n_clusters):
    data = tf_idf_calculator()
    data = data.todense()
    data = StandardScaler().fit_transform(data)
    clustering = AgglomerativeClustering(n_clusters=n_clusters, affinity="cosine",
                                         linkage="complete").fit(data)
    return clusters_dictionary(data, clustering.labels_)
