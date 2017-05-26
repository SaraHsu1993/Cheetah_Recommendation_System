import sys
import string
import pickle
import random
import numpy as np
from scipy import sparse
from sklearn.feature_extraction import DictVectorizer
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(PROJECT_ROOT)

def apps_data_process(user_profile,app_profile):
    user_id_row = {}
    row = 0
    user_profiles = []
    user_apps = []
    user_profile_keys = ["user_model", "user_maxmind_city", "user_maxmind_country_iso_code", "user_country",
                         "user_maxmind_state_iso_code", "user_age", "user_gender"]

    # e.g. www.google.* are all considered to be google
    def filter_keyword(keyword):
        keywords = ["google", "lookout", "twitter", "instagram", "facebook", "t-mobile", "happybits", "motorola"]

        for k in keywords:
            if k in keyword:
                return k

        return keyword

    for user_id in user_profile.keys():
        one_user_profile = {}
        for profile_key in user_profile_keys:
            try:
                one_user_profile[profile_key] = str(user_profile[user_id][profile_key])
            except KeyError:
                pass

        user_profiles.append(one_user_profile)
        #profile_apps_app.append(one_user_profile)
        user_id_row[user_id] = row
        row += 1

        # Use Pkgs and and gp frequency

        try:
            user_pkgs = user_profile[user_id]['user_gp_frequency']
        except KeyError:
            user_pkgs = {}

        if "user_gp_keywords" in user_profile[user_id].keys():
            for user_gp_keyword in user_profile[user_id]['user_gp_keywords'].keys():
                pre_user_gp_keyword = "key_" + filter_keyword(user_gp_keyword)
                user_pkgs[pre_user_gp_keyword] = user_profile[user_id]['user_gp_keywords'][user_gp_keyword]

        user_pkgs = {}
        if 'user_all_pkgs' in user_profile[user_id].keys():
            for app_name in user_profile[user_id]['user_all_pkgs']:
                user_pkgs[app_name] = 1
        user_apps.append(user_pkgs)
        #profile_apps_app.append(user_pkgs)

    installed_apps = []      # information of installed apps
    user_profiles_copy = []  # User profile
    lables = []     # 1-d list of lables, return 1 if exist and 0 if not exits for certain app
    tested_apps = []


    print 'there are %s users'%len(user_apps)
    print 'start'

    #for every user, generate test data set
    for user_index in range(len(user_apps)):
        print 'processing %sth data'%user_index #user index
        # for every installed app generate a test data
        for app_index in user_apps[user_index].keys():
            # for every installed app generate a dict, which prepares for generate spare matrix
            for installed_app in user_apps[user_index].keys():
                if installed_app == app_index:
                    tested_apps.append({installed_app: 1})
                    break
            # for every tested app generate an installed app list
            installed_apps.append(user_apps[user_index])
            # for every tested app generate a lable
            lables.append(1)
            # for every tested app generate user's personal profile
            user_profiles_copy.append(user_profiles[user_index])

        # generate some negative test data that have '-1' lable
        i=0
        while(i<len(user_apps[i].keys())):
            #get an uninstalled app and generate a dict, which prepares for generates spare matrix
            uninstalled_app = user_apps[i].keys()[random.randint(0, len(user_apps[i].keys()) - 1)]
            if uninstalled_app not in user_apps[i].keys():
                i=i+1
                tested_apps.append({uninstalled_app: 1})
                # for every tested app generate an installed app list
                installed_apps.append(user_apps[user_index])
                # for every tested app generate a lable
                lables.append(0)
                # for every tested app generate user's personal profile
                user_profiles_copy.append(user_profiles[user_index])

    print len(installed_apps)
    print len(installed_apps[0])

    print len(user_profiles_copy)
    print len(user_profiles_copy[0])

    print len(tested_apps)
    print len(tested_apps[0])

    print len(lables)

    print "Processing: installed_apps_enc"
    installed_apps_enc = DictVectorizer(sparse=True)
    installed_apps_mat = installed_apps_enc.fit_transform(installed_apps)
    print installed_apps_mat.shape

    print "Processing: user_profiles_copy_enc"
    user_profiles_copy_enc = DictVectorizer(sparse=True)
    user_profiles_copy_mat = user_profiles_copy_enc.fit_transform(user_profiles_copy)
    print user_profiles_copy_mat.shape

    print "Processing: tested_apps"
    tested_apps_enc = DictVectorizer(sparse=True)
    tested_apps_mat = tested_apps_enc.fit_transform(tested_apps)
    print tested_apps_mat.shape

    print "Processing: user_app_test"
    user_app_test = sparse.hstack([installed_apps_mat, user_profiles_copy_mat, tested_apps_mat])
    print user_app_test.shape

    # matrix for user profile are dumped in this file in binary
    pickle.dump([user_app_test], open(os.path.join(BASE_DIR, "user_apps_tl_mat"), "w"))
    pickle.dump([lables], open(os.path.join(BASE_DIR, "user_apps_tl_label_mat"), "w"))

if __name__ == "__main__":
    user_profile = pickle.load(open("user_profile"))
    app_profile = pickle.load(open("app_profile"))
    apps_data_process(user_profile, app_profile)