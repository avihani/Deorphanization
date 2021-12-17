import pandas as pd
import numpy as np
import os

import sklearn
from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold

#Load amino acid dummies
msa_one_hot = pd.read_csv("./../mouseOR_alignment/fasta_60p_aaIdentity_dummies.csv", index_col = 0).transpose()
#Load categorical responses to odors
response_cat_respSub = pd.read_csv("./../mouseOR_alignment/binary_odor_response.csv", index_col = 0).transpose()
#Convert to numpy for faster processing
X = np.array(msa_one_hot)
y = np.array(response_cat_respSub)
olfr_arr = np.array(response_cat_respSub.index)

#Center-scale function for numpy
def np_normalize(input_vector):
    return((input_vector - input_vector.mean())/((input_vector - input_vector.mean()).std(ddof=1)))

param_grid = {'C': [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
              'kernel': ['linear'], 'probability': [True], 'random_state': [42]}

random_seed_init = 42
total_cycles = 100

pred_cols = np.array(['Olfr', 'true', 'prediction', 'cid', 'cycle'])
weights_cols = np.array(['position_aa','weight','cycle','cid','C_regularization'])

all_predictions = pd.DataFrame()
all_weights = pd.DataFrame()

seed_count = 0

for cid in range(response_cat_respSub.shape[1]):
    odor_resp = y[:,cid]
    for cycle in range(total_cycles):
        print("Odor number:", cid+1, ",","iteration:", cycle+1)
        while True:
            X_train, X_test, y_train, y_test, olfr_train, olfr_test = train_test_split(X, odor_resp, olfr_arr,
                                                                                       test_size = 0.1, random_state = random_seed_init+seed_count)
            seed_count = seed_count + 1
            if (y_test.var(ddof = 1) != 0):
                break
        #Identify zero variance column numbers
        zero_var_cols_train = np.where(X_train.var(ddof = 1, axis = 0) == 0)[0]
        zero_var_cols_test = np.where(X_test.var(ddof = 1, axis = 0) == 0)[0]
        zero_var_cols = np.unique(np.concatenate((zero_var_cols_train, zero_var_cols_test)))
        #Drop zero variance features from training/testing/shuffled datasets
        if len(zero_var_cols) > 0:
            X_train = np.delete(X_train, zero_var_cols, axis = 1)
            X_test = np.delete(X_test, zero_var_cols, axis = 1)
        X_train_scaled = np.apply_along_axis(func1d = np_normalize, axis = 0, arr=X_train)
        X_test_scaled = np.apply_along_axis(func1d = np_normalize, axis = 0, arr=X_test)
        optimal_params = GridSearchCV(SVC(),
                                      param_grid,
                                      cv = StratifiedKFold(n_splits=10, shuffle=True,
                                                           random_state=random_seed_init+seed_count),
                                      scoring = 'roc_auc',
                                      refit = True,
                                      n_jobs = 18)
        seed_count = seed_count + 1
        optimal_params.fit(X_train_scaled, y_train)
        y_preds = optimal_params.best_estimator_.predict_proba(X_test_scaled)
        preds = np.hstack([olfr_test.reshape(-1,1),
                           y_test.reshape(-1,1),
                           y_preds[:,1].reshape(-1, 1),
                           np.full((1,olfr_test.shape[0]), response_cat_respSub.columns[cid]).reshape(-1,1),
                           np.full((1,olfr_test.shape[0]), cycle+1).reshape(-1,1)])
        feat_list = np.array(msa_one_hot.drop(msa_one_hot.iloc[:,zero_var_cols].columns, axis = 1).columns)
        coefs = optimal_params.best_estimator_.coef_[0]
        weights = np.hstack([feat_list.reshape(-1,1),
                             coefs.reshape(-1,1),
                             np.full((1,feat_list.shape[0]), cycle+1).reshape(-1,1),
                             np.full((1,feat_list.shape[0]), response_cat_respSub.columns[cid]).reshape(-1,1),
                             np.full((1, feat_list.shape[0]), optimal_params.best_params_['C']).reshape(-1,1)])
        all_predictions = pd.concat([all_predictions, pd.DataFrame(preds)])
        all_weights = pd.concat([all_weights, pd.DataFrame(weights)])


all_predictions.columns = pred_cols
all_weights.columns = weights_cols

all_predictions.to_csv("./../mouseOR_alignment/linear_svc_preds.csv")
all_weights.to_csv("./../mouseOR_alignment/linear_svc_weights.csv")
