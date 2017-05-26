import os
from file_process import file_process
from apps_data_process import apps_data_process
from articles_data_process import articles_data_process
from logistic_regression import logistic_regression
# from tl_app_data_process import tl_app_data_process
import cohtl

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(PROJECT_ROOT)
data_filepath = os.path.join(BASE_DIR, 'ArticleRec/data/part-r-00000-bedbaf06-0f39-43e3-ae2b-8cd18adf551b')
user_profile_filepath = os.path.join(BASE_DIR, 'ArticleRec/user_profile')
app_profile_filepath = os.path.join(BASE_DIR, 'ArticleRec/app_profile')
article_info_filepath = os.path.join(BASE_DIR, 'ArticleRec/article_info')
#train_data_tl_filepath = os.path.join(BASE_DIR, 'ArticleRec/')

if __name__ == "__main__":

    print 'processing raw data'
    file_process(data_filepath)
    print 'Done'

    print 'processing app data'
    apps_data_process(user_profile_filepath, app_profile_filepath)
    print 'Done'
    
    print 'running LR model for app recommendation'
    train_data_app_filepath = os.path.join(BASE_DIR, 'ArticleRec/encoded_app_data')
    logistic_regression(train_data_app_filepath)
    print 'Done'

    print 'processing article data'
    articles_data_process(user_profile_filepath, article_info_filepath, app_profile_filepath)
    print 'Done'

    print 'running LR model for article recommendation'
    train_data_article_filepath = os.path.join(BASE_DIR, 'ArticleRec/encoded_onlyarticle_data')
    logistic_regression(train_data_article_filepath)
    print 'Done'

    # print 'processing app data for TL'
    # tl_app_data_process(user_profile_filepath, app_profile_filepath)
    # print 'Done'

    # print 'feature selecting for TL'
    # cohtl()
    # print 'running LR model for TL'
    #logistic_regression(train_data_tl_filepath)