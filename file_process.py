import ast
import pickle


def file_process(data_filepath):
    user_profile = {}
    app_profile = {}
    article_info = {}
    with open(data_filepath) as f:
        data = f.readlines()
        article_count = 1
        for line in data:
            record = ast.literal_eval(line.strip())
            user_id = record['user_id']
            user_profile_keys = ['user_age', 'user_gender', 'user_gender_weight', 'user_model', 'user_brand',
                                 'user_country', 'user_maxmind_country_iso_code', 'user_maxmind_state_iso_code',
                                 'user_maxmind_city']
            app_profile_keys = ['user_all_pkgs', 'user_gp_frequency', 'user_gp_keywords']
            article_info_keys = ['article_contentid', 'article_l1_categories', 'article_l2_categories',
                                 'article_dwelltime', 'article_lan', 'article_publisher', 'article_title',
                                 'article_update_time', 'dt']
            if user_id in user_profile.keys():
                article_count += 1
                article_info[user_id][article_count] = {}
                for key in article_info_keys:
                    if key in record.keys():
                        article_info[user_id][article_count][key] = record[key]
            else:
                article_count = 1
                user_profile[user_id] = {}
                app_profile[user_id] = {}
                article_info[user_id] = {}
                article_info[user_id][article_count] = {}
                for key in user_profile_keys:
                    if key in record.keys():
                        user_profile[user_id][key] = record[key]
                for key in app_profile_keys:
                    if key in record.keys():
                        app_profile[user_id][key] = record[key]
                for key in article_info_keys:
                    if key in record.keys():
                        temp = record[key]
                        if key == 'article_title':
                            temp = temp.lower().strip()
                        if key == 'article_publisher':
                            temp = temp.lower()
                        if key == 'article_lan' and 'en' in temp:
                            temp = 'en'
                        article_info[user_id][article_count][key] = temp

    pickle.dump(user_profile, open('user_profile', 'w'))
    pickle.dump(app_profile, open('app_profile', 'w'))
    pickle.dump(article_info, open('article_info', 'w'))