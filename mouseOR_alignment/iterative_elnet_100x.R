library(glmnet)
library(tidyverse)
library(doParallel)

registerDoParallel(30)

#Load categorical responses of Olfr to odors
response_cat_respSub <- t(read.delim("./../mouseOR_alignment/binary_odor_response.csv", header=T, sep=',', row.names = 1))
#Load one hot encoded alignment
one_hot_msa <- t(read.delim("./../mouseOR_alignment/fasta_60p_aaIdentity_dummies.csv", sep=',', header=T, row.names=1))
#Load grid searched alpha parameter
parameters_lst <- readRDS("./../mouseOR_alignment/alpha_search_logisticRegression_out.rds")

#Identify optimal parameters from grid search
parameters_out <- as.data.frame(do.call("rbind", parameters_lst))
optimal_params <- parameters_out %>%
    group_by(odor) %>%
    slice(which.min(cvm.min))
    
#Build and test models on ligand responsive ORs
set.seed(42)

model_predictions_lst <- list()
model_min_weights <- list()
model_1se_weights <- list()

fold <- 10
test_size <- floor(dim(response_cat_respSub)[1]/fold)

cycle <- 100

for (cid in colnames(response_cat_respSub)){
    #Create odor id name
    odor_id <- paste0('odor_',cid)
    #Setup lists
    model_predictions_lst[[odor_id]] <- data.frame()
    model_min_weights[[odor_id]] <- data.frame()
    model_1se_weights[[odor_id]] <- data.frame()
    for (cycle_round in 1:cycle){
        print(paste(odor_id, cycle_round))
        repeat{
            test_indices <- sample(1:dim(response_cat_respSub)[1], test_size)
            ytest_data <- response_cat_respSub[test_indices,cid]
            if (var(ytest_data) != 0) {break}
        }
        #Create x/y data
        x <- one_hot_msa
        y <- factor(response_cat_respSub[,cid])
        #Create train data
        xtrain <- x[-test_indices,]
        ytrain <- y[-test_indices]
        #Create test data
        xtest <- x[test_indices,]
        ytest <- y[test_indices]
        #Remove zero variance features after splitting into test/train
        if (length(which(apply(xtrain, 2, var) == 0)) > 0){
            xtrain_updated <- xtrain[,-(which(apply(xtrain, 2, var) == 0))]
            xtest_updated <- xtest[,-(which(apply(xtrain, 2, var) == 0))]
        } else {
            xtrain_updated <- xtrain
            xtest_updated <- xtest
        }
        if (length(which(apply(xtest_updated, 2, var) == 0)) > 0){
            xtrain_updated2 <- xtrain_updated[,-(which(apply(xtest_updated,2,var) == 0))]
            xtest_updated2 <- xtest_updated[,-(which(apply(xtest_updated,2,var) == 0))]
        } else {
            xtrain_updated2 <- xtrain_updated
            xtest_updated2 <- xtest_updated
        }
        #Scale train/test
        xtrain_updated2 <- scale(xtrain_updated2, center=TRUE, scale=TRUE)
        xtest_updated2 <- scale(xtest_updated2, center=TRUE, scale=TRUE)
        #Fit model
        model_fit <- cv.glmnet(x=xtrain_updated2, y=ytrain,
                               parallel=TRUE,
                               family = "binomial",
                               alpha = optimal_params[optimal_params$odor == cid,]$alpha,
                               nfolds = 10)
        #Use model to predict test data
        model_predict_min <- predict(model_fit, s = model_fit$lambda.min, newx = xtest_updated2, type = "response")
        model_predict_1se <- predict(model_fit, s = model_fit$lambda.1se, newx = xtest_updated2, type = "response")
        #Compile predictions
        prediction_out <- cbind(names(ytest), as.matrix(ytest), min_pred = as.numeric(model_predict_min), se1_pred = as.numeric(model_predict_1se), cid, cycle_round)
        colnames(prediction_out) <- c("receptor_id","ytest","lambda_min_pred","lambda_1se_pred","odor_id","iteration")
        row.names(prediction_out) <- NULL
        model_predictions_lst[[odor_id]] <- rbind.data.frame(model_predictions_lst[[odor_id]], prediction_out)
        #Pull lambda.1se cost function values/data
        lambda_1se_val <- model_fit$lambda.1se
        lambda_1se_index <- which(model_fit$lambda == lambda_1se_val)
        weights_1se <- data.frame(position_aa = names(model_fit$glmnet.fit$beta[,lambda_1se_index]),
                                  weights = model_fit$glmnet.fit$beta[,lambda_1se_index],
                                  cid = cid, cycle = cycle_round)
        row.names(weights_1se) <- NULL
        model_1se_weights[[odor_id]] <- rbind.data.frame(model_1se_weights[[odor_id]], weights_1se)
        #Pull lambda.min cost function values/data    
        lambda_min_val <- model_fit$lambda.min
        lambda_min_index <- which(model_fit$lambda == lambda_min_val)
        weights_min <- data.frame(position_aa = names(model_fit$glmnet.fit$beta[,lambda_min_index]),
                                  weights = model_fit$glmnet.fit$beta[,lambda_min_index],
                                  cid = cid, cycle = cycle_round)
        row.names(weights_min) <- NULL
        model_min_weights[[odor_id]] <- rbind.data.frame(model_min_weights[[odor_id]], weights_min)
    }
}


saveRDS(model_min_weights, "./../mouseOR_alignment/elnet_min_weights_100x.rds")
saveRDS(model_1se_weights, "./../mouseOR_alignment/elnet_1se_weights_100x.rds")
saveRDS(model_predictions_lst, "./../mouseOR_alignment/elnet_preds_100x.rds")
