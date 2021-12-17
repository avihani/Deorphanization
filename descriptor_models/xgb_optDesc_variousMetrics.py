import pandas as pd
import numpy as np

import itertools

from scipy.spatial.distance import pdist

import xgboost as xgb

#Center-scale function
def normalize(x):
    return (x - x.mean())/(x - x.mean()).std()

#Center-scale function for numpy
def np_normalize(input_vector):
    return((input_vector - input_vector.mean())/((input_vector - input_vector.mean()).std(ddof=1)))

#Calculate distances between features
def feature_distances(input_vector):
    modified_vector = input_vector.reshape(-1,1)
    vector_distances = pdist(modified_vector, 'euclidean')
    return vector_distances

#Center-scale function for shuffled data
def scale_shuffled(input_vector):
    if input_vector.var() == 0:
        input_vector.values[:] = 0
    else:
        input_vector = input_vector.transform(normalize)
    return input_vector

#Load data
well_predicted_desc = pd.read_csv("./../compiled_desc_resp/well_predicted_desc.csv", index_col = 0)
tested_odor_resp = pd.read_csv("./../compiled_desc_resp/compiled_odor_sigResp_wide.csv", index_col = 0)
tested_odor_desc = well_predicted_desc.copy(deep=True)
#Normalize data
norm_tested_odor_desc = tested_odor_desc.transform(normalize)
norm_tested_odor_resp = tested_odor_resp.transform(normalize)
#Load shuffled data
shuffled_odor_desc = pd.read_csv("./../compiled_desc_resp/shuffled_filtered_tested_optDesc.csv", index_col = 0)
shuffled_odor_desc.index = tested_odor_desc.index
#Normalize shuffled data, no columns display numerical instabilities
norm_shuffled_odor_desc = shuffled_odor_desc.transform(scale_shuffled)
norm_shuffled_odor_desc.columns = norm_tested_odor_desc.columns

#Convert base data to numpy array for faster processing
tested_odor_desc = np.array(tested_odor_desc)
tested_odor_resp = np.array(tested_odor_resp)
shuffled_odor_desc = np.array(shuffled_odor_desc)
#Convert normalized data to numpy array for faster processing
norm_tested_odor_desc = np.array(norm_tested_odor_desc)
norm_tested_odor_resp = np.array(norm_tested_odor_resp)
norm_shuffled_odor_desc = np.array(norm_shuffled_odor_desc)

xgb_predictions = pd.DataFrame()

for i, (o1, o2) in enumerate(itertools.combinations(range(tested_odor_desc.shape[0]), 2)):
    print(i)
    temp_predictions = pd.DataFrame()
    #Setup/reset models
    Euc_model = xgb.XGBRegressor(random_state = 42, n_jobs = -1)
    cor_model = xgb.XGBRegressor(random_state = 42, n_jobs = -1)
    cos_model = xgb.XGBRegressor(random_state = 42, n_jobs = -1)
    #Setup testing data
    norm_testing_desc = norm_tested_odor_desc[[o1,o2]]
    norm_testing_resp = norm_tested_odor_resp[[o1, o2]]
    shuffled_testing_desc = norm_shuffled_odor_desc[[o1, o2]]
    #Setup training data
    training_desc = np.delete(tested_odor_desc, [o1,o2], axis = 0)
    training_resp = np.delete(tested_odor_resp, [o1,o2], axis = 0)
    #Remove features with zero variance in training data
    if (training_desc.var(ddof = 1, axis = 0) == 0).any():
        #Identify zero variance column numbers
        zero_var_cols = np.where((training_desc.var(ddof = 1,axis = 0) == 0))
        #Drop zero variance features from training/testing/shuffled datasets
        training_desc = np.delete(training_desc, zero_var_cols, axis = 1)
        norm_testing_desc = np.delete(norm_testing_desc, zero_var_cols, axis = 1)
        shuffled_testing_desc = np.delete(shuffled_testing_desc, zero_var_cols, axis = 1)
    #Normalize training data in absence of test set
    norm_training_desc = np.apply_along_axis(func1d = np_normalize, axis = 0, arr=training_desc)
    norm_training_resp = np.apply_along_axis(func1d = np_normalize, axis = 0, arr=training_resp)
    #Calculate descriptor distances in train and test set
    norm_training_desc_dist = np.apply_along_axis(func1d = feature_distances, axis = 0, arr=norm_training_desc)
    norm_testing_desc_dist = np.apply_along_axis(func1d = feature_distances, axis = 0, arr=norm_testing_desc)
    shuffled_desc_dist = np.apply_along_axis(func1d = feature_distances, axis = 0, arr=shuffled_testing_desc)
    #Calculate response distances
    norm_training_resp_dist_Euc = pdist(norm_training_resp, 'euclidean')
    norm_testing_resp_dist_Euc = pdist(norm_testing_resp, 'euclidean')
    #Calculate correlation response distances
    norm_training_resp_dist_cor = pdist(norm_training_resp, 'correlation')
    norm_testing_resp_dist_cor = pdist(norm_testing_resp, 'correlation')
    #Calculate cosine response distances
    norm_training_resp_dist_cos = pdist(norm_training_resp, 'cosine')
    norm_testing_resp_dist_cos = pdist(norm_testing_resp, 'cosine')
    #Fit models
    Euc_model.fit(norm_training_desc_dist, norm_training_resp_dist_Euc)
    cor_model.fit(norm_training_desc_dist, norm_training_resp_dist_cor)
    cos_model.fit(norm_training_desc_dist, norm_training_resp_dist_cos)
    #Use models to predict intact distances
    ypred_Euc = Euc_model.predict(norm_testing_desc_dist)
    ypred_cor = cor_model.predict(norm_testing_desc_dist)
    ypred_cos = cos_model.predict(norm_testing_desc_dist)
    #Use models to predict shuffled distances
    ypred_Euc_shuffled = Euc_model.predict(shuffled_desc_dist)
    ypred_cor_shuffled = cor_model.predict(shuffled_desc_dist)
    ypred_cos_shuffled = cos_model.predict(shuffled_desc_dist)
    #Compile predictions
    temp_predictions = pd.concat([pd.DataFrame(norm_testing_resp_dist_Euc),
                                  pd.DataFrame(norm_testing_resp_dist_cor),
                                  pd.DataFrame(norm_testing_resp_dist_cos),
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
#Calculate full fit response distances
norm_tested_odor_resp_dist_Euc = pd.DataFrame(pdist(norm_tested_odor_resp, 'euclidean'))
norm_tested_odor_resp_dist_cor = pd.DataFrame(pdist(norm_tested_odor_resp, 'correlation'))
norm_tested_odor_resp_dist_cos = pd.DataFrame(pdist(norm_tested_odor_resp, 'cosine'))
#Calculate full fit feature Euclidean distances
norm_tested_odor_desc_dist = np.apply_along_axis(func1d = feature_distances, axis = 0, arr=norm_tested_odor_desc)
#Fit models
Euc_model.fit(norm_tested_odor_desc_dist, norm_tested_odor_resp_dist_Euc)
cor_model.fit(norm_tested_odor_desc_dist, norm_tested_odor_resp_dist_cor)
cos_model.fit(norm_tested_odor_desc_dist, norm_tested_odor_resp_dist_cos)
#Use full fit models to make predictions
full_fit_pred_Euc = pd.DataFrame(Euc_model.predict(norm_tested_odor_desc_dist))
full_fit_pred_cor = pd.DataFrame(cor_model.predict(norm_tested_odor_desc_dist))
full_fit_pred_cos = pd.DataFrame(cos_model.predict(norm_tested_odor_desc_dist))
#Compile full fit
full_fit_out = pd.concat([norm_tested_odor_resp_dist_Euc,
                          norm_tested_odor_resp_dist_cor,
                          norm_tested_odor_resp_dist_cos,
                          full_fit_pred_Euc,
                          full_fit_pred_cor,
                          full_fit_pred_cos], axis = 1)
full_fit_out.columns = ['fullFit_ytrue_Euc','fullFit_ytrue_cor','fullFit_ytrue_cos',
                        'fullFit_ypred_Euc','fullFit_ypred_cor','fullFit_ypred_cos']
#Save files
xgb_predictions.to_csv("./../descriptor_models/xgb_LOOCV_optDesc_variousMetrics.csv")
full_fit_out.to_csv("./../descriptor_models/xgb_fullFit_optDesc_variousMetrics.csv")

