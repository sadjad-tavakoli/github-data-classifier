import sys

from topic_modeling.topic_modeling import find_model_number, topic_modeling, mallet_topic_modeling

if __name__ == "__main__":
    request = sys.argv[1] if len(sys.argv) > 1 else "built-in"
    n_topics = sys.argv[2] if len(sys.argv) > 2 else "11"

    if request == "finding":
        find_model_number()
    elif request == "mallet" and n_topics.isdigit():
        mallet_topic_modeling(int(n_topics))
    elif request.isdigit():
        topic_modeling(int(request))
    elif request == "built-in" and n_topics.isdigit():
        topic_modeling(int(n_topics))
    else:
        print("\n invalid request")
