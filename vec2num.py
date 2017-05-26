import string
import pickle
import random
import numpy as np
from scipy import sparse
from sklearn.feature_extraction import DictVectorizer
from sklearn.datasets import dump_svmlight_file

def clean_title(line):
    identify = string.maketrans('', '')
    delEStr = string.punctuation + string.digits
    aline = line.decode("ascii", "ignore").encode("ascii")
    cleanLine = aline.translate(identify, delEStr)

    return cleanLine

def article_dict2num(article_content):
    article_id_row = {}
    row = 0
    article_content_str = []
    article_bog = []
    article_names = []

    article_content_keyword = ['article_publisher', 'article_lan']

    for article_id in article_content.keys():
        one_article_content = {}
        for key in article_content_keyword:
            try:
                one_article_content[key] = article_content[article_id][key]
            except KeyError:
                pass

        # L1 and L2categories
        if 'article_l1_categories' in article_content[article_id].keys():
            for l1 in article_content[article_id]['article_l1_categories']:
                one_article_content['l1_' + l1['name']] = l1['weight']
        if 'article_l2_categories' in article_content[article_id].keys():
            for l2 in article_content[article_id]['article_l2_categories']:
                one_article_content['l2_' + l2['name']] = l2['weight']

        try:
            article_names.append(article_content[article_id]['article_title'])
            article_bog.append(dict((word.lower(), 1) for word in
                                    clean_title(article_content[article_id]['article_title'].strip()).split()))
        except KeyError:
            article_names.append("NA")
            article_bog.append({})

        article_content_str.append(one_article_content)
        article_id_row[article_id] = row
        row += 1

    article_enc = DictVectorizer(sparse=True)
    article_mat = article_enc.fit_transform(article_content_str)

    article_bog_enc = DictVectorizer(sparse=True)
    article_bog_mat = article_bog_enc.fit_transform(article_bog)
    #print len(article_bog_mat)

    # matrix for article content (do not contain title) and matrix for article title(using bag of word as features) are stored in these two files
    pickle.dump([article_id_row, article_enc.get_feature_names(), article_mat],
                open("article_content_mat.pkl", "w"))
    pickle.dump([article_id_row, article_bog_enc.get_feature_names(), article_bog_mat, article_names],
                open("article_bog.pkl", "w"))


if __name__ == "__main__":
    article_content = pickle.load(open("article_content.pkl"))
    article_dict2num(article_content)