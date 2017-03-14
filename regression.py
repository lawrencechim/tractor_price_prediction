from sklearn.linear_model import LinearRegression
from clean_data import clean_train, clean_test
import numpy as np
import pandas as pd
import sys

def rmlse(y_true, y_pred):
    y_pred = y_pred + 1
    y_true = y_true + 1
    error = np.square(np.log(y_pred) - np.log(y_true)).sum()
    root_log_error = np.sqrt(error/len(y_true))
    return root_log_error

if __name__ == '__main__':
    filename = 'data/train.csv'

    X, y, feat_cols, dummies, mode, encl_dummies, mons, auc_id =  clean_train(filename)

    model = LinearRegression()
    model.fit(X, y)

    testfile = 'data/test.csv'
    X_test, salesids = clean_test(testfile, feat_cols, dummies,
            mode, encl_dummies, mons, auc_id)

    y_pred = model.predict(X_test)

    out_dict = pd.DataFrame({'SalesID': salesids,
                'SalePrice': y_pred}, columns = ['SalesID', 'SalePrice'])

    output_file = 'data/our_predictions_' + str(sys.argv[1]) + '.csv'
    out_dict.to_csv(output_file, header = ['SalesID', 'SalePrice'],
        index = False)

    # print "Current score: {:.5f}".format(rmlse(y_test, y_pred))
