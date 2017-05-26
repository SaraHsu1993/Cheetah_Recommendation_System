import pickle
import datetime
from scipy import sparse
from sklearn.feature_extraction import DictVectorizer
import os
from sklearn.datasets import dump_svmlight_file
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(PROJECT_ROOT)
from collections import Counter
def articles_data_process(user_profile_filepath, article_info_filepath, app_profile_filepath):
    user = pickle.load(open(user_profile_filepath, 'r'))
    article = pickle.load(open(article_info_filepath, 'r'))
    # apps = pickle.load(open(app_profile_filepath, 'r'))

    # titles=[]
    # for userid in article.keys():
    #     for article_index in article[userid].keys():
    #         for article_feature in article[userid][article_index].keys():
    #             if 'article_title' == article_feature:
    #                 titles.append(article[userid][article_index]['article_title'])
    #
    # valid_article = dict(Counter(titles))
    #
    # for title in valid_article.keys():
    #     if valid_article[title] < 2:
    #         del valid_article[title]

    lables = []
    user_ids = []
    userid = 0
    user_articles = []
    user_profiles = []

    user_gp_frequency = []
    user_pkgs = []

    article_keys = ['article_lan', 'article_publisher', 'article_title', 'article_update_time', 'dt']

    for user_id in user.keys():
        print 'processing %dth user article data' %userid
        for article_index in article[user_id].keys():
            check = False
            if 'article_l1_categories' in article[user_id][article_index].keys():
                check = True
                for index in range(len(article[user_id][article_index]['article_l1_categories'])):
                    temp = {}
                    for key in article_keys:
                        #for key in valid_article:
                        if key in article[user_id][article_index].keys():
                            if key == 'article_update_time':
                                temp[key] = datetime.datetime.fromtimestamp(
                                    article[user_id][article_index][key]).strftime('%Y%m%d')
                            else:
                                temp[key] = article[user_id][article_index][key]
                    temp['category_name'] = article[user_id][article_index]['article_l1_categories'][index]['name']
                    temp['category_weight'] = article[user_id][article_index]['article_l1_categories'][index]['weight']
                    if 'article_dwelltime' in article[user_id][article_index].keys():
                        ### if article[user_id][article_index]['article_dwelltime'] > 3000:
                            ### delete record
                        if article[user_id][article_index]['article_dwelltime'] > 30 and article[user_id][article_index]['article_dwelltime'] < 3000:
                            lables.append(1)
                        else:
                            lables.append(0)
                    else:
                        lables.append(0)
                    user_ids.append({userid: 1})
                    user_articles.append(temp)
                    user_profiles.append(user[user_id])
                    # if 'user_gp_frequency' in apps[user_id].keys():
                    #     user_gp_frequency.append(apps[user_id]['user_gp_frequency'])
                    # else:
                    #     user_gp_frequency.append({})
                    # temp = {}
                    # if 'user_all_pkgs' in apps[user_id].keys():
                    #     for app_name in apps[user_id]['user_all_pkgs']:
                    #         temp[app_name] = 1
                    # user_pkgs.append(temp)
            if 'article_l2_categories' in article[user_id][article_index].keys():
                check = True
                for index in range(len(article[user_id][article_index]['article_l2_categories'])):
                    temp = {}
                    for key in article_keys:
                        if key in article[user_id][article_index].keys():
                            if key == 'article_update_time':
                                temp[key] = datetime.datetime.fromtimestamp(
                                    article[user_id][article_index][key]).strftime('%Y%m%d')
                            else:
                                temp[key] = article[user_id][article_index][key]
                    temp['category_name'] = article[user_id][article_index]['article_l2_categories'][index]['name']
                    temp['category_weight'] = article[user_id][article_index]['article_l2_categories'][index]['weight']
                    if 'article_dwelltime' in article[user_id][article_index].keys():
                        if article[user_id][article_index]['article_dwelltime'] > 30:
                            lables.append(1)
                        else:
                            lables.append(0)
                    else:
                        lables.append(0)
                    user_ids.append({userid: 1})
                    user_articles.append(temp)
                    # user_profiles.append(user[user_id])
                    # if 'user_gp_frequency' in apps[user_id].keys():
                    #     user_gp_frequency.append(apps[user_id]['user_gp_frequency'])
                    # else:
                    #     user_gp_frequency.append({})
                    # temp = {}
                    # if 'user_all_pkgs' in apps[user_id].keys():
                    #     for app_name in apps[user_id]['user_all_pkgs']:
                    #         temp[app_name] = 1
                    # user_pkgs.append(temp)
            if not check:
                for key in article_keys:
                    if key in article[user_id][article_index].keys():
                        if key == 'article_update_time':
                            temp[key] = datetime.datetime.fromtimestamp(
                                article[user_id][article_index][key]).strftime('%Y%m%d')
                        else:
                            temp[key] = article[user_id][article_index][key]
                if 'article_dwelltime' in article[user_id][article_index].keys():
                    if article[user_id][article_index]['article_dwelltime'] > 30:
                        lables.append(1)
                    else:
                        lables.append(0)
                else:
                    lables.append(0)
                user_ids.append({userid: 1})
                user_articles.append(temp)
                user_profiles.append(user[user_id])
                # if 'user_gp_frequency' in apps[user_id].keys():
                #     user_gp_frequency.append(apps[user_id]['user_gp_frequency'])
                # else:
                #     user_gp_frequency.append({})
                # temp = {}
                # if 'user_all_pkgs' in apps[user_id].keys():
                #     for app_name in apps[user_id]['user_all_pkgs']:
                #         temp[app_name] = 1
                # user_pkgs.append(temp)
        userid += 1

    user_profiles_enc = DictVectorizer()
    user_profiles_mat = user_profiles_enc.fit_transform(user_profiles)
    print "User profile shape: ", user_profiles_mat.shape

    user_articles_enc = DictVectorizer()
    user_articles_mat = user_articles_enc.fit_transform(user_articles)
    print "Article shape: ", user_articles_mat.shape

    user_ids_enc = DictVectorizer()
    user_ids_mat = user_ids_enc.fit_transform(user_ids)
    print "User id shape: ", user_ids_mat.shape

    # user_gp_frequency_enc = DictVectorizer()
    # user_gp_frequency_mat = user_gp_frequency_enc.fit_transform(user_gp_frequency)
    # print user_gp_frequency_mat.shape
    #
    # user_pkgs_enc = DictVectorizer()
    # user_pkgs_mat = user_pkgs_enc.fit_transform(user_pkgs)
    # print user_pkgs_mat.shape

    ### user_articles_test = sparse.hstack([user_ids_mat, user_profiles_mat, user_articles_mat])
    user_articles_test = sparse.hstack([user_ids_mat, user_articles_mat])
    # user_articles_test = sparse.hstack([user_ids_mat, user_articles_mat])

    print "User-article test shape: ", user_articles_test.shape

    pickle.dump([user_articles_test, lables], open('encoded_onlyarticle_data', 'w'))

    print "Separating dataset "
    data_shape = user_articles_test.shape
    row_num = data_shape[0]
    train_num = int(row_num * 0.8)

    fm_data_csr = user_articles_test.tocsr()
    fm_train_X = fm_data_csr[:train_num]
    fm_test_X = fm_data_csr[train_num + 1:]
    fm_train_Y = lables[:train_num]
    fm_test_Y = lables[train_num + 1:]

    fm_train_path = os.path.join(BASE_DIR, "ArticleRec/libfm-1.40.src/article_train_mat.pkl")
    fm_test_path = os.path.join(BASE_DIR, "ArticleRec/libfm-1.40.src/article_test_mat.pkl")
    dump_svmlight_file(fm_train_X, fm_train_Y, fm_train_path)
    dump_svmlight_file(fm_test_X, fm_test_Y, fm_test_path)
    fm_train_lables = os.path.join(BASE_DIR, "ArticleRec/libfm-1.40.src/article_fm_train_lable.txt")
    with open(fm_train_lables, 'w') as f:
        for i in range(0,train_num):
            f.write(str(lables[i]) + '\n')
    fm_test_lables = os.path.join(BASE_DIR, "ArticleRec/libfm-1.40.src/article_fm_test_lable.txt")
    with open(fm_test_lables, 'w') as f:
        for i in range(train_num + 1, row_num):
            f.write(str(lables[i]) + '\n')