library(tidyverse)
library(glmnet)
library(doParallel)
library(ggpubr)
library(scales)

registerDoParallel(30)

#Open the data files
input_receptor_resp_wide <- read.delim("./../compiled_desc_resp/compiled_odor_sigResp_wide.csv", sep=',', header=T, row.names = 1)
input_odor_descs_wide <- read.delim("./../compiled_desc_resp/filtered_tested_desc.csv", sep=',', header=T, row.names = 1)
#Sort data
input_odor_descs_wide <- input_odor_descs_wide[sort(row.names(input_odor_descs_wide)),]
input_receptor_resp_wide <- input_receptor_resp_wide[sort(row.names(input_receptor_resp_wide)),]
#Create a matrix of all possible pairwise combinations of the different odors
odor_perms  <- t(combn(row.names(input_odor_descs_wide), 2))

#Feature Euclidean distance
calc_vec_dist <- function(input_vector){
    vect_dist <- as.matrix(dist(input_vector))
    vect_dist[lower.tri(vect_dist, diag=T)] <- NA
    vect_dist <- na.omit(reshape2::melt(vect_dist))
    return(vect_dist$value)
}

#Response Euclidean distance
calc_resp_distance <- function(A) {
    square_dist <- as.matrix(dist(A))
    square_dist[lower.tri(square_dist, diag=T)] <- NA
    square_dist <- na.omit(reshape2::melt(square_dist))
    return(square_dist$value)
}

#Response correlation distance
calc_resp_corr_dist <- function(A){
    square_corr_dist <- as.matrix(1-cor(t(A)))
    square_corr_dist[lower.tri(square_corr_dist, diag=T)] <- NA
    square_corr_dist <- na.omit(reshape2::melt(square_corr_dist))
    return(square_corr_dist$value)
}

#Response cosine distance
calc_resp_cos_dist <- function(A){
    Matrix <- as.matrix(A)
    sim <- Matrix / sqrt(rowSums(Matrix * Matrix))
    sim <- sim %*% t(sim)
    sim <- 1-sim
    sim[lower.tri(sim, diag=T)] <- NA
    sim <- na.omit(reshape2::melt(sim))
    return(sim$value)
}

#Create function to resample with replacement
resample_with_replacement <- function(input_vector){
    resampled_vector <- input_vector[sample(length(input_vector), replace=TRUE)]
    return(resampled_vector)
}

#Function to scale resampled data
scale_resample <- function(input_vector){
    if (var(input_vector) == 0){
        input_vector <- rep(0, length(input_vector))
    } else {
        input_vector <- scale(input_vector, center=TRUE, scale=TRUE)
    }
    return(input_vector)
}

set.seed(42)

#Can run locally
lambda_array <- seq(from = 0, to = 3, length = 1000)

#Normalize the data
norm_odor_descs_wide <- scale(input_odor_descs_wide, center=TRUE, scale=TRUE)
norm_receptor_resp_wide <- scale(input_receptor_resp_wide, center=TRUE, scale=TRUE)

#Generate fictitious odors whose features are drawn from
#the distribution of features from the tested odors
shuffled_odor_descs_wide <- apply(input_odor_descs_wide, 2, resample_with_replacement)
#Normalize the shuffled data
norm_shuffled_odor_descs_wide <- apply(shuffled_odor_descs_wide, 2, scale_resample)
row.names(norm_shuffled_odor_descs_wide) <- row.names(input_odor_descs_wide)

write.csv(shuffled_odor_descs_wide, "./../compiled_desc_resp/shuffled_filtered_tested_desc.csv")

regression_model_Euc <- list()
regression_model_cor <- list()
regression_model_cos <- list()
lambdaArray_out <- list()
zero_var_feats <- data.frame()

for (i in seq(1:nrow(odor_perms))){
    print(i)
    #Select test odors
    odor1 <- odor_perms[i,1]
    odor2 <- odor_perms[i,2]
    odor_list <- c(odor1, odor2)
    odor_set <- paste0("CID","_",odor1,"_",odor2)
    #Select test rows
    test_odors_rows <- which(row.names(input_odor_descs_wide) %in% odor_list)
    #Setup the descriptors data
    train_odors_desc <- input_odor_descs_wide[-test_odors_rows,]
    test_odors_desc <- norm_odor_descs_wide[test_odors_rows,]
    shuffled_test_odors_desc <- norm_shuffled_odor_descs_wide[test_odors_rows,]
    #Setup the response data
    train_odors_resp <- input_receptor_resp_wide[-test_odors_rows,]
    test_odors_resp <- norm_receptor_resp_wide[test_odors_rows,]
    #Find features with zero variance 
    zero_var_cols <- which(apply(train_odors_desc, 2, var) == 0)
    if (length(zero_var_cols) > 0) {
        tempZero <- data.frame(i, odor1, odor2, zero_var_cols, names(zero_var_cols))
        zero_var_feats <- rbind(zero_var_feats, tempZero)
        #Remove features with zero variance
        zero_var_cols <- as.numeric(zero_var_cols)
        train_odors_desc <- train_odors_desc[,-zero_var_cols]
        test_odors_desc <- test_odors_desc[,-zero_var_cols]
        shuffled_test_odors_desc <- shuffled_test_odors_desc[,-zero_var_cols]
    }
    #Center/scale the train data
    train_odors_desc <- scale(train_odors_desc, center=TRUE, scale=TRUE)
    train_odors_resp <- scale(train_odors_resp, center=TRUE, scale=TRUE)
    #Training dataset distance calculation
    xtrain <- apply(train_odors_desc, 2, calc_vec_dist)
    ytrain_Euc <- as.matrix(suppressWarnings(calc_resp_distance(train_odors_resp)))
    ytrain_cor <- as.matrix(suppressWarnings(calc_resp_corr_dist(train_odors_resp)))
    ytrain_cos <- as.matrix(suppressWarnings(calc_resp_cos_dist(train_odors_resp)))
    #Testing dataset distance calculation
    xtest <- t(apply(test_odors_desc, 2, dist))
    shuffled_xtest <- t(apply(shuffled_test_odors_desc, 2, dist))
    ytest_Euc <- as.matrix(suppressWarnings(calc_resp_distance(test_odors_resp)))
    ytest_cor <- as.matrix(suppressWarnings(calc_resp_corr_dist(test_odors_resp)))
    ytest_cos <- as.matrix(suppressWarnings(calc_resp_cos_dist(test_odors_resp)))
    #Select shuffled test
    #Train the linear model
    linear_model_Euc <- glmnet(xtrain, ytrain_Euc,
                               parallel=TRUE,
                               type.measure="mse", 
                               alpha=1,
                               lambda = lambda_array,
                               family="gaussian")
    linear_model_cor <- glmnet(xtrain, ytrain_cor,
                               parallel=TRUE,
                               type.measure="mse", 
                               alpha=1,
                               lambda = lambda_array,
                               family="gaussian")
    linear_model_cos <- glmnet(xtrain, ytrain_cos,
                               parallel=TRUE,
                               type.measure="mse", 
                               alpha=1,
                               lambda = lambda_array,
                               family="gaussian")
    regression_model_Euc[[odor_set]] <- linear_model_Euc
    regression_model_cor[[odor_set]] <- linear_model_cor
    regression_model_cos[[odor_set]] <- linear_model_cos
    #Make predictions on intact test data
    new_y <- data.frame()
    ypred_Euc <- predict(linear_model_Euc, s = lambda_array, newx=xtest, type = "response")
    ypred_cor <- predict(linear_model_cor, s = lambda_array, newx=xtest, type = "response")
    ypred_cos <- predict(linear_model_cos, s = lambda_array, newx=xtest, type = "response")
    #Make predictions on shuffled test data
    ypred_shuffled_Euc <- predict(linear_model_Euc, s = lambda_array, newx=shuffled_xtest, type = "response")
    ypred_shuffled_cor <- predict(linear_model_cor, s = lambda_array, newx=shuffled_xtest, type = "response")
    ypred_shuffled_cos <- predict(linear_model_cos, s = lambda_array, newx=shuffled_xtest, type = "response")
    #Compile datasets
    temp <- cbind.data.frame(ytest_Euc = rep(ytest_Euc), ytest_cor = rep(ytest_cor), ytest_cos = rep(ytest_cos),
                             ypred_Euc = t(ypred_Euc), ypred_cor = t(ypred_cor), ypred_cos = t(ypred_cos),
                             ypred_shuffled_Euc = t(ypred_shuffled_Euc), ypred_shuffled_cor = t(ypred_shuffled_cor), ypred_shuffled_cos = t(ypred_shuffled_cos),
                             lambda = lambda_array,
                             coeff_Euc = rev(linear_model_Euc$df), coeff_cor = rev(linear_model_cor$df), coeff_cos = rev(linear_model_cos$df),
                             odors = rep(odor_set))
    temp$ypred_Euc_mse <- (temp$ytest_Euc - temp$ypred_Euc)^2
    temp$ypred_cor_mse <- (temp$ytest_cor - temp$ypred_cor)^2
    temp$ypred_cos_mse <- (temp$ytest_cos - temp$ypred_cos)^2
    temp$ypred_shuffled_Euc_mse <- (temp$ytest_Euc - temp$ypred_shuffled_Euc)^2
    temp$ypred_shuffled_cor_mse <- (temp$ytest_cor - temp$ypred_shuffled_cor)^2
    temp$ypred_shuffled_cos_mse <- (temp$ytest_cos - temp$ypred_shuffled_cos)^2
    lambdaArray_out[[odor_set]] <- temp
    }

saveRDS(lambdaArray_out, "./LASSO_odorCV_predictions_all_metrics.rds")
saveRDS(regression_model_Euc, "./LASSO_odorCV_Euc_models.rds")
saveRDS(regression_model_cor, "./LASSO_odorCV_cor_models.rds")
saveRDS(regression_model_cos, "./LASSO_odorCV_cos_models.rds")
