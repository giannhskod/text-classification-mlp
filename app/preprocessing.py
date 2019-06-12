import os
import itertools
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from gensim.models.wrappers import FastText

from app import DATA_DIR

DATASET_FILENAME = 'stack-overflow-data.csv'
EMBEDDINGS_FILENAME = 'cc.en.300.bin'

def text_centroid(text, model):
    text_vec = []
    counter = 0
    sent_text = nltk.sent_tokenize(text)
    for sentence in sent_text:
        sent_tokenized = nltk.word_tokenize(sentence)
        for word in sent_tokenized:
            try:
                if counter == 0:
                    text_vec = model[word.lower()]
                else:
                    text_vec = np.add(text_vec, model[word.lower()])
                counter += 1
            except:
                pass

    return np.asarray(text_vec) / counter


def load_dataset(tags_categories='__all__'):
    """
    Loads and returns the Dataset as a DataFrame.

    :param tags_categories: If '__all__' is given then all the dataset is parsed.
                    Otherwise a list of the class names should be passed that
                    will filter the dataset against.
    :return: A DataFrame filled with the whole or a subset of the dataset loaded.

    """

    assert tags_categories == '__all__' or isinstance(tags_categories, list)  or isinstance(tags_categories, tuple), \
        ("Argument <tags_categories> should be a type of 'list' or 'tuple' or a string with explicit value '__all__'."
         "Instead it got the value {}".format(tags_categories))
    dataset_path = os.path.join(DATA_DIR, DATASET_FILENAME)
    dataset_df = pd.read_csv(dataset_path)
    dataset_df['tags'] = pd.Categorical(dataset_df.tags)
    return dataset_df if tags_categories == '__all__' else dataset_df.loc[dataset_df['tags'].isin(tags_categories)]


def preprocess_data(input_data, label_field, text_field, input_ins='as_tf_idf', **kwargs):
    """
    Generates the train-test data for the model based on the given arguments.

    Args:
        'input_data' (pandas.DataFrame): The dataset Dataframe that will be splitted in train-test data.
        'input_ins' (str): If 'as_tf_idf' is given then it generates tf-idf vectors per text row
                     If 'as_centroids' is given then it generates centroids per text row
    It returns a dictionary with the below structure:
        {
            'x_train' (numpy.array),
            'x_test' (numpy.array),
            'y_train' (numpy.array),
            'y_test' (numpy.array),
        }

    """
    assert all([label_field, text_field]), \
        "Fields <label_field>, <text_field> cannot be None or empty"
    train, test = train_test_split(input_data, test_size=0.3, random_state=1596)

    x_train = np.array(list(itertools.chain.from_iterable(train[[text_field]].values.astype('U').tolist())))
    x_test = np.array(list(itertools.chain.from_iterable(test[[text_field]].values.astype('U').tolist())))

    if input_ins == 'as_tf_idf':
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000, sublinear_tf=True,
                                     stop_words=stopwords.words('english'))

        x_train = vectorizer.fit_transform(x_train).toarray()
        x_test = vectorizer.transform(x_test).toarray()

    else:

        embeddings = kwargs.get('embeddings')
        if not embeddings:
            import ipdb
            ipdb.set_trace()
            embeddings_path = os.path.join(DATA_DIR, EMBEDDINGS_FILENAME)
            embeddings = FastText.load_fasttext_format(embeddings_path)
        x_train = np.array(list(map(lambda text: text_centroid(text, embeddings), x_train)))
        x_train = np.stack(x_train, axis=0)
        x_test = np.array(list(map(lambda text: text_centroid(text, embeddings), x_test)))
        x_test = np.stack(x_test, axis=0)

    mlb = MultiLabelBinarizer()

    y_train = mlb.fit_transform(train[[label_field]].values.tolist())
    y_test = mlb.transform(test[[label_field]].values.tolist())

    return {
        'x_train': x_train,
        'x_test': x_test,
        'y_train': y_train,
        'y_test': y_test
    }


if __name__ == "__main__":
    data = load_dataset()
    print(data['tags'].categories)
