import pandas as pd
import numpy as np

import itertools

from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

from sklearn.preprocessing import StandardScaler

from scipy.spatial.distance import pdist

input_resp = pd.read_csv("./../compiled_desc_resp/compiled_odor_sigResp_wide.csv", index_col = 0)
input_desc = pd.read_csv("./../ohe_features/optimized_desc_svc_ohe.csv", index_col = 0)

#Calculate distances between features
def feature_distances(input_vector):
    modified_vector = input_vector.reshape(-1,1)
    vector_distances = pdist(modified_vector, 'euclidean')
    return vector_distances


resampled_odors = pd.read_csv("./../ohe_features/shuffle_optimized_desc_svc_ohe.csv", index_col = 0)
resampled_odors.columns = input_desc.columns

#Convert to numpy for faster processing
input_desc_arr = np.array(input_desc)
input_resp_arr = np.array(input_resp)
resampled_odors_arr = np.array(resampled_odors)

xgb_predictions = pd.DataFrame()

for i, (o1, o2) in enumerate(itertools.combinations(range(input_desc.shape[0]), 2)):
    print(i)
    temp_predictions = pd.DataFrame()
    #Setup/reset models
    Euc_model = xgb.XGBRegressor(random_state = 42, n_jobs = -1)
    cor_model = xgb.XGBRegressor(random_state = 42, n_jobs = -1)
    cos_model = xgb.XGBRegressor(random_state = 42, n_jobs = -1)
    #Setup testing data
    x_test = input_desc_arr[[o1,o2]]
    shuffle_test = resampled_odors_arr[[o1,o2]]
    y_test = input_resp_arr[[o1,o2]]
    #Setup training data
    x_train = np.delete(input_desc_arr, [o1,o2], axis = 0)
    y_train = np.delete(input_resp_arr, [o1,o2], axis = 0)
    #Scale response data
    resp_scaler = StandardScaler()
    resp_scaler.fit(y_train)
    norm_y_train = resp_scaler.transform(y_train)
    norm_y_test = resp_scaler.transform(y_test)
    #Scale descriptor data
    desc_scaler = StandardScaler()
    desc_scaler.fit(x_train)
    norm_x_train = desc_scaler.transform(x_train)
    norm_x_test = desc_scaler.transform(x_test)
    norm_shuffle_test = desc_scaler.transform(shuffle_test)
    #Calculate descriptor distances in train and test set
    x_train_dist = np.apply_along_axis(func1d = feature_distances, axis = 0, arr=norm_x_train)
    x_test_dist = np.apply_along_axis(func1d = feature_distances, axis = 0, arr=norm_x_test)
    shuffle_test_dist = np.apply_along_axis(func1d = feature_distances, axis = 0, arr=norm_shuffle_test)
    #Calculate response distances
    y_train_Euc_dist = pdist(norm_y_train, 'euclidean')
    y_test_Euc_dist = pdist(norm_y_test, 'euclidean')
    #Calculate correlation response distances
    y_train_cor_dist = pdist(norm_y_train, 'correlation')
    y_test_cor_dist = pdist(norm_y_test, 'correlation')
    #Calculate cosine response distances
    y_train_cos_dist = pdist(norm_y_train, 'cosine')
    y_test_cos_dist = pdist(norm_y_test, 'cosine')
    #Fit models
    Euc_model.fit(x_train_dist, y_train_Euc_dist)
    cor_model.fit(x_train_dist, y_train_cor_dist)
    cos_model.fit(x_train_dist, y_train_cos_dist)
    #Use models to predict intact distances
    ypred_Euc = Euc_model.predict(x_test_dist)
    ypred_cor = cor_model.predict(x_test_dist)
    ypred_cos = cos_model.predict(x_test_dist)
    #Use models to predict shuffled distances
    ypred_Euc_shuffled = Euc_model.predict(shuffle_test_dist)
    ypred_cor_shuffled = cor_model.predict(shuffle_test_dist)
    ypred_cos_shuffled = cos_model.predict(shuffle_test_dist)
    #Compile predictions
    temp_predictions = pd.concat([pd.DataFrame(y_test_Euc_dist),
                                  pd.DataFrame(y_test_cor_dist),
                                  pd.DataFrame(y_test_cos_dist),
                                  pd.DataFrame(ypred_Euc),
                                  pd.DataFrame(ypred_cor),
                                  pd.DataFrame(ypred_cos),
                                  pd.DataFrame(ypred_Euc_shuffled),
                                  pd.DataFrame(ypred_cor_shuffled),
                                  pd.DataFrame(ypred_cos_shuffled)],
                                  axis = 1)
    xgb_predictions = pd.concat([xgb_predictions, temp_predictions], axis = 0)

xgb_predictions.columns = ['ytrue_Euc','ytrue_cor','ytrue_cos',
                           'ypred_Euc','ypred_cor','ypred_cos',
                           'ypred_Euc_shuffled','ypred_cor_shuffled','ypred_cos_shuffled'
                           ]

#Create new models for full fitting evaluation
Euc_model = xgb.XGBRegressor(random_state = 42, n_jobs = -1)
cor_model = xgb.XGBRegressor(random_state = 42, n_jobs = -1)
cos_model = xgb.XGBRegressor(random_state = 42, n_jobs = -1)
#Normalize response data
full_resp_scaler = StandardScaler()
full_resp_scaler.fit(input_resp_arr)
norm_input_resp_arr = full_resp_scaler.transform(input_resp_arr)
#Normalize descriptor data
full_desc_scaler = StandardScaler()
full_desc_scaler.fit(input_desc_arr)
norm_input_desc_arr = full_desc_scaler.transform(input_desc_arr)
#Calculate full fit response distances
input_resp_arr_Euc_dist = pd.DataFrame(pdist(norm_input_resp_arr, 'euclidean'))
input_resp_arr_cor_dist = pd.DataFrame(pdist(norm_input_resp_arr, 'correlation'))
input_resp_arr_cos_dist = pd.DataFrame(pdist(norm_input_resp_arr, 'cosine'))
#Calculate full fit feature Euclidean distances
norm_input_desc_arr_dist = np.apply_along_axis(func1d = feature_distances, axis = 0, arr=norm_input_desc_arr)
#Fit models
Euc_model.fit(norm_input_desc_arr_dist, input_resp_arr_Euc_dist)
cor_model.fit(norm_input_desc_arr_dist, input_resp_arr_cor_dist)
cos_model.fit(norm_input_desc_arr_dist, input_resp_arr_cos_dist)
#Use full fit models to make predictions
full_fit_pred_Euc = pd.DataFrame(Euc_model.predict(norm_input_desc_arr_dist))
full_fit_pred_cor = pd.DataFrame(cor_model.predict(norm_input_desc_arr_dist))
full_fit_pred_cos = pd.DataFrame(cos_model.predict(norm_input_desc_arr_dist))
#Compile full fit
full_fit_out = pd.concat([input_resp_arr_Euc_dist,
                          input_resp_arr_cor_dist,
                          input_resp_arr_cos_dist,
                          full_fit_pred_Euc,
                          full_fit_pred_cor,
                          full_fit_pred_cos], axis = 1)
full_fit_out.columns = ['fullFit_ytrue_Euc','fullFit_ytrue_cor','fullFit_ytrue_cos',
                        'fullFit_ypred_Euc','fullFit_ypred_cor','fullFit_ypred_cos']

#Save files
xgb_predictions.to_csv("./../ohe_features/xgb_LOOCV_optDesc_variousMetrics.csv")
full_fit_out.to_csv("./../ohe_features/xgb_fullFit_optDesc_variousMetrics.csv")

