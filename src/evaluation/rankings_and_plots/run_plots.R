# runs functions to create plots
# source files "calc_ranking.R", "load_data.R" and "plot_ranking.R" first and set paths

# simulated data
# caution: takes a few seconds to create plot!
path <- "raw_results/simulation/"
data <- load_data(path, dataset_names = paste0("S", rep(c(1,2), each = 3), letters[1:3]))
df <- calc_ranking(data)
p <- plot_ranking(df)
plot(p)

# bcw dataset
path <- "raw_results/results_BCW/"
data <- load_data(path, dataset_names = "BCW", outlier_metric = "f1")
df <- calc_ranking(data)
p <- plot_ranking(df)
plot(p)

# servo dataset
path <- "raw_results/results_servo/"

data <- load_data(path, dataset_names = "servo")
df <- calc_ranking(data)
p <- plot_ranking(df)
plot(p)

