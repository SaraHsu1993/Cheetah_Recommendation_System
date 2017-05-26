import pickle
import random
import os
from scipy import sparse
from sklearn.feature_extraction import DictVectorizer
from sklearn.datasets import dump_svmlight_file
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(PROJECT_ROOT)

def apps_data_process(user_profile_filepath, app_profile_filepath):
    user = pickle.load(open(user_profile_filepath, 'r'))
    apps = pickle.load(open(app_profile_filepath, 'r'))

    user_profiles = []
    user_gp_frequency = []
    user_pkgs = []
    lables = []
    tested_apps = []
    user_ids = []
    user_profiles_copy = []
    gp_frequency_copy = []

    for user_id in user.keys():
        user_profiles.append(user[user_id])
        if 'user_gp_frequency' in apps[user_id].keys():
            user_gp_frequency.append(apps[user_id]['user_gp_frequency'])
        else:
            user_gp_frequency.append({})
        temp = {}
        if 'user_all_pkgs' in apps[user_id].keys():
            for app_name in apps[user_id]['user_all_pkgs']:
                temp[app_name] = 1
        user_pkgs.append(temp)

    user_pkgs_enc = DictVectorizer()
    user_pkgs_mat = user_pkgs_enc.fit_transform(user_pkgs)
    pkgs_names = user_pkgs_enc.get_feature_names()

    for user_index in range(len(user_pkgs)):
        for app_name in user_pkgs[user_index].keys():
            tested_apps.append({app_name: 1})
            lables.append(1)
            user_profiles_copy.append(user_profiles[user_index])
            user_ids.append({user_index: 1})
            gp_frequency_copy.append(user_gp_frequency[user_index])

        i=0
        while(i<len(user_pkgs[user_index].keys())):
            uninstalled_app = pkgs_names[random.randint(0, len(pkgs_names) - 1)]
            if uninstalled_app not in user_pkgs[user_index].keys():
                i=i+1
                tested_apps.append({uninstalled_app: 1})
                lables.append(0)
                user_profiles_copy.append(user_profiles[user_index])
                user_ids.append({user_index: 1})
                gp_frequency_copy.append(user_gp_frequency[user_index])

    user_profiles_copy_enc = DictVectorizer()
    user_profiles_copy_mat = user_profiles_copy_enc.fit_transform(user_profiles_copy)
    print user_profiles_copy_mat.shape

    tested_apps_enc = DictVectorizer()
    tested_apps_mat = tested_apps_enc.fit_transform(tested_apps)
    print tested_apps_mat.shape

    user_ids_enc = DictVectorizer()
    user_ids_mat = user_ids_enc.fit_transform(user_ids)
    print user_ids_mat.shape

    gp_frequency_copy_enc = DictVectorizer()
    gp_frequency_copy_mat = gp_frequency_copy_enc.fit_transform(gp_frequency_copy)
    print gp_frequency_copy_mat.shape

    user_app_test = sparse.hstack([user_ids_mat, user_profiles_copy_mat, tested_apps_mat, gp_frequency_copy_mat])
    # user_app_test = sparse.hstack([user_ids_mat, tested_apps_mat, gp_frequency_copy_mat])
    print "shape of user_app", user_app_test.shape


    pickle.dumo([user_app_test].open('','w'))
    pickle.dump([user_app_test, lables], open('encoded_app_data', 'w'))
    #pickle.dump([user_app_test, lables], open('encoded_onlyapp_data', 'w'))

    print "Separating dataset "
    data_shape = user_app_test.shape
    row_num = data_shape[0]
    train_num = int(row_num * 0.8)

    fm_data_csr = user_app_test.tocsr()
    fm_train_X = fm_data_csr[:train_num]
    fm_test_X = fm_data_csr[train_num + 1:]
    fm_train_Y = lables[:train_num]
    fm_test_Y = lables[train_num + 1:]

    fm_train_path = os.path.join(BASE_DIR, "ArticleRec/libfm-1.40.src/app_train_mat.pkl")
    fm_test_path = os.path.join(BASE_DIR, "ArticleRec/libfm-1.40.src/app_test_mat.pkl")
    dump_svmlight_file(fm_train_X, fm_train_Y, fm_train_path)
    dump_svmlight_file(fm_test_X, fm_test_Y, fm_test_path)
    fm_lables = os.path.join(BASE_DIR, "ArticleRec/libfm-1.40.src/app_fm_test_lable.txt")
    with open(fm_lables, 'w') as f:
        for i in range(train_num + 1, row_num):
            f.write(str(lables[i]) + '\n')