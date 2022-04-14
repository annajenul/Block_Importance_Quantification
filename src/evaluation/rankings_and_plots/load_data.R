load_data <- function(path, dataset_names, outlier_metric = "r2"){
  require(dplyr)
  
  # list files and extract setup and method names
  list <- list.files(paste0(path), 
                     full.names = TRUE, 
                     recursive = TRUE, 
                     include.dirs = FALSE)
  names_all <- sapply(strsplit(list, "/"), unlist)
  res_ind <- grepl("MI", names_all[nrow(names_all),]) | grepl("vargrad", names_all[nrow(names_all),]) # files containing results
  perf_ind <- grepl(outlier_metric, names_all[nrow(names_all),]) # files containing performance information
  names <- names_all[,res_ind]
  if(length(dataset_names) > 1){
    setup = as.factor(names[nrow(names) - 1,])
  } else{
    setup = as.factor(dataset_names)
  }
  names <- data.frame(setup = setup,
                      method = as.factor(names[nrow(names),]))
  levels(names$setup) <- dataset_names
  levels(names$method) <- c("knock-in", "knock-out", "vargrad-max", "vargrad-mean")
  
  # outlier runs
  outlier_runs <- data.frame(setup = c(), run = c())
  for(i in 1:sum(perf_ind)){
    print(unlist(read.csv(list[perf_ind][i]))[unlist(read.csv(list[perf_ind][i])) < 0.8])
    outlier_run <- which(unlist(read.csv(list[perf_ind][i])) < 0.8)
    if(length(outlier_run) > 0){
      outlier_runs <- rbind(
        outlier_runs, data.frame(
          setup = levels(names$setup)[i],
          run = outlier_run
        )
      )
    }
  }
  print(outlier_runs %>% group_by(setup) %>% summarize(n = n()))
  
  # read files and aggregate in data.frame
  data <- c()
  for(i in 1:nrow(names)){
    newdata <- read.csv(list[res_ind][i])
    outliers <- subset(outlier_runs, setup == names[i, "setup"])$run
    if(!is.null(outliers)){
      newdata <- newdata[-outliers,]
    }
    
    suppressWarnings(
      data <- rbind(
        data,
        cbind(
          names[i,],
          newdata
        )
      )
    )
  }
  colnames(data) = c("setup", "method", paste0("B", 1 : (ncol(data) - 2)))
  
  return(data)
}
