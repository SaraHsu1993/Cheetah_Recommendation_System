# import string
# import pickle
# import random
# import numpy as np
# from scipy import sparse
# from sklearn.feature_extraction import DictVectorizer
# from sklearn.datasets import dump_svmlight_file
#
# def entire_set(user_article, article_content):
#     user_id = record[]
#
#
# def sort_article(sort_user_article, sort_article_content):
#     #merge user_id with all article_contents
#
#     #sort by td
#
#     #filter useless records (dwell time too short || article appears too few times)
#
#     #one-hot-encode || auto-encode
#
#     pickle.dump([article_id_row, article_enc.get_feature_names(), article_mat],
#                 open("article_train.pkl", "w"))
#     pickle.dump([article_id_row, article_enc.get_feature_names(), article_mat],
#                 open("article_test.pkl", "w"))
#
# if __name__=="__main__":
#     user_article = pickle.load(open("user_article.pkl"))
#     article_content = pickle.load(open("article_content.pkl"))
#
#     entire_set(user_article, article_content)
#
#     sort_user_article = entire_set().sort_user_article
#     sort_article_content = entire_set().sort_article_content
#     sort_article(sort_user_article, sort_article_content)