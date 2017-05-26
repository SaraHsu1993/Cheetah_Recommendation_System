# import numpy as np
# import pandas as pd
#
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import roc_auc_score
# import pickle
# from scipy.sparse import coo_matrix
#
# # create dataframes with an intercept column and dummy variables for
# # occupation and occupation_husb
# train_data = pickle.load(open("user_apps_train_mat.pkl"))
# test_data = pickle.load(open("user_apps_test_mat.pkl"))
#
# label = np.array(train_data[1])
# X = train_data[0].tocsr()
#
#
#
# regularization = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
# for i in regularization:
#     print i
#         # instantiate a logistic regression model, and fit with X and y
#         model = LogisticRegression(C=i)
#         model = model.fit(tr_X, tr_label)
#
#         # check the accuracy/AUC on the training set
#         tr_AUC = roc_auc_score(tr_label, model.predict(tr_X))
#         tst_AUC = roc_auc_score(tst_label, model.predict(tst_X))
#         # tr_acc = np.mean(model.predict(tr_X) == tr_label)
#         # tst_acc = np.mean(model.predict(tst_X) == tst_label)
#
#         print "%dth Fold Train AUC:%f, Test AUC: %f" % (
#             cv, tr_AUC, tst_AUC
#             # cv, tr_acc, tst_acc
#         )

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.cross_validation import KFold
import pickle
from scipy.sparse import coo_matrix

# create dataframes with an intercept column and dummy variables for
# occupation and occupation_husb
#train_data = pickle.load(open("user_apps_train_mat.pkl"))
# train_data = pickle.load(open("user_apps_train_mat_nogp.pkl"))
train_data = pickle.load(open("/Users/xuyujie/Desktop/article-pro/encoded_onlyarticle_data"))

label = np.array(train_data[1])
X = train_data[0].tocsr()

regularization = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
for i in regularization:
    print i
    # 5 Fold Cross Validation
    kf = KFold(n=len(label), n_folds=5, shuffle=True)

    cv = 0
    for tr, tst in kf:
        # Train Test Split
        tr_X = X[tr, :]
        tr_label = label[tr]

        tst_X = X[tst, :]
        tst_label = label[tst]

        # instantiate a logistic regression model, and fit with X and y
        model = LogisticRegression(C=i)
        model = model.fit(tr_X, tr_label)

        # check the accuracy/AUC on the training set
        tr_AUC = roc_auc_score(tr_label, model.predict(tr_X))
        tst_AUC = roc_auc_score(tst_label, model.predict(tst_X))
        # tr_acc = np.mean(model.predict(tr_X) == tr_label)
        # tst_acc = np.mean(model.predict(tst_X) == tst_label)

        print "%dth Fold Train AUC:%f, Test AUC: %f" % (
            cv, tr_AUC, tst_AUC
            # cv, tr_acc, tst_acc
        )
        cv += 1

