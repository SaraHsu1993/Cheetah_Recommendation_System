import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.cross_validation import KFold


def logistic_regression(train_data_filepath):
    train_data = pickle.load(open(train_data_filepath, 'r'))

    label = np.array(train_data[1])
    x = train_data[0].tocsr()

    kf = KFold(n=len(label), n_folds=5, shuffle=True)
    cv = 0
    for tr, tst in kf:
        tr_x = x[tr, :]
        tr_label = label[tr]

        tst_x = x[tst, :]
        tst_label = label[tst]

        model = LogisticRegression()
        model = model.fit(tr_x, tr_label)

        tr_auc = roc_auc_score(tr_label, model.predict(tr_x))
        tst_auc = roc_auc_score(tst_label, model.predict(tst_x))

        print "%dth Fold Train AUC:%f, Test AUC: %f" % (
            cv, tr_auc, tst_auc
        )
        cv += 1