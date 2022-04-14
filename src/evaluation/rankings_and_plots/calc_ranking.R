# calculate ranking based on input data
calc_ranking <- function(data, scale = TRUE, pval = 0.01){
  require(dplyr)
  
  blocknames <- colnames(data)[-c(1,2)]
  setupnames <- unique(data$setup)
  methodnames <- unique(data$method)
  
  nblocks <- length(blocknames)
  nsetups <- length(setupnames)
  nmethods <- length(methodnames)
  
  # build data.frame
  df <- data.frame( method = data$method,
                    setup = data$setup,
                    val = unlist(data[,-c(1,2)]),
                    block = rep(colnames(data[,-c(1,2)]), each = nrow(data))
  )
  
  # min-max scaling by method & setup
  if(scale){
    scale_this <- function(x){
      (x - min(x, na.rm = TRUE)) / (max(x, na.rm = TRUE) - min(x, na.rm = TRUE))
    }
    df <- 
      df %>%
      group_by(method, setup) %>%
      mutate(val = scale_this(val))
  }

  # pairwise Wilcoxon test
  pwt <- array(dim = c(nsetups, nmethods, nblocks, nblocks))
  dimnames(pwt)[[1]] <- setupnames
  dimnames(pwt)[[2]] <- methodnames
  dimnames(pwt)[[3]] <- dimnames(pwt)[[4]] <- blocknames
  
  # iterate over setups & methods
  summary <- c()  
  for(i in setupnames){
    for(j in methodnames){
      s <- subset(df, setup == i & method == j)
      
      # calculate block-wise means & initial ranking
      mean <- (s %>% group_by(block) %>% summarize(mean = mean(val)))$mean
      ranking = rank(-mean, ties.method = "min")
      
      # pairwise Wilcoxon-test (gives warning)
      suppressWarnings(
        pwt[i,j,-1,-dim(pwt)[4]] <- pairwise.wilcox.test(x = s$val,
                                                       g = s$block,
                                                       p.adjust.method = "bonf",
                                                       paired = TRUE)$p.value > pval
      )
      for(b1 in 1:nblocks){
        for(b2 in setdiff(1:nblocks, (b1:nblocks))){
          # check if pairwise difference is non-significant
          if(!is.na(pwt[i,j,b1,b2]) & pwt[i,j,b1,b2]){
            # update ranking
            ranking[ranking == ranking[b1] | ranking == ranking[b2]] <- min(c(ranking[b1], ranking[b2]))
          }
        }
      }
      
      # add to summary list
      summary <- rbind(summary,
                    data.frame(
                       setup = i,
                       method = j,
                       block = unique(s$block),
                       mean = mean,
                       rank = ranking
                    )
      )
    }
  }
  
  # add to original data frame
  df <- inner_join(df, summary, by = c("setup", "method", "block"))
  
  return(df)
}
