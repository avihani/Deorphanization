library(glmnet)
library(tidyverse)
library(doParallel)

registerDoParallel(30)

#Load categorical responses of Olfr to odors
response_cat_respSub <- t(read.delim("./../mouseOR_alignment/binary_odor_response.csv", header=T, sep=',', row.names = 1))
#Load one hot encoded alignment
one_hot_msa <- t(read.delim("./../mouseOR_alignment/fasta_60p_aaIdentity_dummies.csv", sep=',', header=T, row.names=1))

#Alpha grid search
#Takes a while
set.seed(42)

foldid <- sample(1:10,size=dim(one_hot_msa)[1],replace=TRUE)

alpha_search <- seq(1,99,1)

alpha_list <- list()
parameters_lst <- list()

xtrain <- one_hot_msa
xtrain <- scale(xtrain, center = TRUE, scale = TRUE)


for (cid in colnames(response_cat_respSub)){
    odor_id <- paste0('odor_',cid)

    ytrain <- factor(response_cat_respSub[,cid])
    
    model_list <- list()
    ind_parameters <- NULL
    
    for (alpha_level in alpha_search) {
        alpha_value <- paste0('alpha_',alpha_level/100)
        print(paste(odor_id,alpha_value))
        glm.model <- cv.glmnet(x = xtrain, y = ytrain,
                               parallel = TRUE,
                               family="binomial",
                               foldid=foldid,
                               alpha=alpha_level/100,
                               nfolds = 10)
        model_list[[alpha_value]] <- glm.model
        ind_parameters <- rbind(ind_parameters, c(alpha_level/100,
                                                  glm.model$lambda.min,
                                                  min(glm.model$cvm),
                                                  glm.model$cvsd[which.min(glm.model$cvm)]))
    }
    ind_parameters <- cbind(ind_parameters, rep(cid))
    colnames(ind_parameters) <- c("alpha","lambda.min","cvm.min","cvsd","odor")
    alpha_list[[odor_id]] <- model_list
    parameters_lst[[odor_id]] <- ind_parameters
}

saveRDS(alpha_list, "./../mouseOR_alignment/alpha_search_logisticRegression.rds")
saveRDS(parameters_lst, "./../mouseOR_alignment/alpha_search_logisticRegression_out.rds")

