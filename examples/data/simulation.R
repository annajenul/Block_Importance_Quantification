library(mvtnorm)
library(clusterGeneration)
library(ggplot2)

setwd("~/GitHub/Block_Importance_Quantification/examples/data/simulation/test")

feature_generator = function(train_size=10000, test_size=50000, n_feats_block, seed = 42){
  # The function generates a dataset with features from a multivariate normal distribution. The block-splitting must be done afterwards by the user.
  # train_size ... number of train samples <integer>
  # test_size ... number of test samples <integer>
  # n_feats_block ... number of features per block <vector>
  # seed ... reproducibility <integer>
  
  set.seed(seed)
  sample_size = train_size + test_size
  feature_size = sum(n_feats_block)
  cov = genPositiveDefMat(feature_size)$Sigma
  X = mvtnorm::rmvnorm(sample_size, mean = rep(0,feature_size), sigma = cov)
  return(X)
}

# common parameters
train_size = 10000
test_size = 10000

target_generator = function(data, 
                            n_feats_block, 
                            n_imp_feats_block = NULL, 
                            imp_feat_weights = NULL, 
                            seed = 42){
  # Generate the target and beta values (ground truth) based on the generated dataset .
  # data ... dataset with features 
  # n_feats_block ... number of features per block in data <vector>
  # n_imp_feats_block ... number of important features per block in data <vector>
  # imp_feat_weights ... coefficients (matrix) of important features <matrix>
  # linear ... linear or non-linear connection between target and data <boolean>
  # seed ... reproducibility <integer>
  
  if(!is.null(imp_feat_weights)){
    if(!is.matrix(imp_feat_weights[[1]])){
      imp_feat_weights = lapply(imp_feat_weights, diag)
    }
    n_imp_feats_block = sapply(imp_feat_weights, nrow)
  }
  print(n_imp_feats_block)
  set.seed(seed)
  beta = matrix(0, nrow = ncol(data), ncol = ncol(data))
  
  block_ind = 1
  for (i in 1:length(n_feats_block)) {
    if(n_imp_feats_block[i]>0){
      print(block_ind : (block_ind + n_imp_feats_block[i] - 1))
      beta[block_ind : (block_ind + n_imp_feats_block[i] - 1) , block_ind : (block_ind + n_imp_feats_block[i] - 1)] = imp_feat_weights[[i]]
    }
    block_ind = block_ind + n_feats_block[i]
  }
  
  y = apply(data, 1, function(s){return(t(s) %*% beta %*% s)})
  return(list(y=y+rnorm(length(y), 0, 0.01 * sd(y)), beta=beta))
}


n_feats_per_block = rep(32,8)
X = feature_generator(train_size = train_size, test_size = test_size, n_feats_block = n_feats_per_block)
repl = unlist(rep(list(1:32, 33:64, 65:96, 97:128), each = 2))

# define the different Setups

S = list(
  S1a = list(diag(rep(7,2)),diag(rep(6,2)),diag(rep(5,2)),diag(rep(4,2)),
            diag(rep(3,2)),diag(rep(2,2)),diag(rep(1,2)),matrix(nrow=0, ncol = 1)),
  
  S1b = list(diag(rep(2,7)),diag(rep(2,6)),diag(rep(2,5)), diag(rep(2,4)),
            diag(rep(2,3)),diag(rep(2,2)),cbind(2),matrix(nrow=0, ncol = 1)),


  S1c = list(cbind(7),diag(rep(6,2)),diag(rep(5,3)),diag(rep(4,4)),diag(rep(3,5)),
            diag(rep(2,6)),diag(rep(1,7)),matrix(nrow=0, ncol = 1)),

  S2a = list(cbind(c(7,1),c(0,7)),
            cbind(c(6,1),c(0,6)),
            cbind(c(5,1),c(0,5)),
            cbind(c(4,1),c(0,4)),
            cbind(c(3,1),c(0,3)),
            cbind(c(2,1),c(0,2)),
            cbind(c(1,1),c(0,1)),
            cbind(c(0,1),c(0,0))), 
  
  S2b = list(cbind(c(2,1,1,1,1,1,1),c(0,2,1,1,1,1,1),c(0,0,2,1,1,1,1),c(0,0,0,2,1,1,1),c(0,0,0,0,2,1,1),c(0,0,0,0,0,2,1),c(0,0,0,0,0,0,2)),
            cbind(c(2,1,1,1,1,1),c(0,2,1,1,1,1),c(0,0,2,1,1,1),c(0,0,0,2,1,1),c(0,0,0,0,2,1),c(0,0,0,0,0,2)),
            cbind(c(2,1,1,1,1),c(0,2,1,1,1),c(0,0,2,1,1),c(0,0,0,2,1),c(0,0,0,0,2)),
            cbind(c(2,1,1,1),c(0,2,1,1),c(0,0,2,1),c(0,0,0,2)),
            cbind(c(2,1,1),c(0,2,1),c(0,0,2)),
            cbind(c(2,1),c(0,2)),
            cbind(2),
            matrix(nrow=0, ncol = 1)),



  S2c = list(cbind(7),
            cbind(c(6,1),c(0,6)),
            cbind(c(5,1,1),c(0,5,1),c(0,0,5)),
            cbind(c(4,1,1,1),c(0,4,1,1),c(0,0,4,1),c(0,0,0,4)),
            cbind(c(3,1,1,1,1),c(0,3,1,1,1),c(0,0,3,1,1),c(0,0,0,3,1),c(0,0,0,0,3)),
            cbind(c(2,1,1,1,1,1),c(0,2,1,1,1,1),c(0,0,2,1,1,1),c(0,0,0,2,1,1),c(0,0,0,0,2,1),c(0,0,0,0,0,2)),
            cbind(c(1,1,1,1,1,1,1),c(0,1,1,1,1,1,1),c(0,0,1,1,1,1,1),c(0,0,0,1,1,1,1),c(0,0,0,0,1,1,1),c(0,0,0,0,0,1,1),c(0,0,0,0,0,0,1)),
            cbind(c(0,1,1,1,1,1,1,1),c(0,0,1,1,1,1,1,1),c(0,0,0,1,1,1,1,1),c(0,0,0,0,1,1,1,1),c(0,0,0,0,0,1,1,1),c(0,0,0,0,0,0,1,1),c(0,0,0,0,0,0,0,1), rep(0,8))
  )
)

write.csv(X[(test_size+1):nrow(X),], file=paste0("X_train.csv"))
write.csv(X[1:test_size,], file=paste0("X_test.csv"))

for(j in 1:length(S)){
  print(paste0("***",j))
  setup = S[[j]]

  tg = target_generator(X, n_feats_block = n_feats_per_block, 
                        imp_feat_weights = setup,
                        seed = 42)
  write.csv(tg$y[(test_size+1):nrow(X)], file=paste0("y_train_", names(S)[j],".csv"))
  write.csv(tg$y[1:test_size], file=paste0("y_test_", names(S)[j],".csv"))
  write.csv(tg$beta, file=paste0("beta_", names(S)[j],".csv"))
}
