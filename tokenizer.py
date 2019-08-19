import gensim
from nltk import WordNetLemmatizer
from nltk.stem.snowball import EnglishStemmer

# nltk.download('wordnet')

stemmer = EnglishStemmer()


# for lemmatization there are some others libraries. I used this one because of\
#  its easy installation and usability for the result reproduction on other systems.
def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))


def text_processing(text):
    result = []
    stop_words = frozenset(
        ["issu", "issue", "issuecom", "github", "test", "thank", "file", "code"]).union(
        gensim.parsing.preprocessing.STOPWORDS)

    for token in gensim.utils.simple_preprocess(text):
        if token not in stop_words:
            stemmed = lemmatize_stemming(token)
            if stemmed not in stop_words:
                result.append(stemmed)
    return result
