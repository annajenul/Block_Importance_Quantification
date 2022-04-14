# plot feature ranking & scores
plot_ranking <- function(df){
  require(ggplot2)
  require(RColorBrewer)
  
  # define color palette (remove lightest color)
  max_rank <- max(df$rank)
  pal <- brewer.pal(max_rank + 1, "Blues")[(max_rank + 1) : 2]
  
  # sort blocks in decreasing order
  df$block <- factor(df$block, levels = sort(unique(df$block), decreasing = TRUE))
  df$method <- factor(df$method, levels = c("vargrad-max", "vargrad-mean", "knock-in", "knock-out"))
  
  # create plot
  p <- ggplot(df, aes(x = val, y = block, fill = ordered(rank))) + 
    geom_density_ridges(color = "black", size = 0.3, alpha = 0.8) + 
    geom_label(aes(x = -0.15, y = block, label = rank), color = "white", size = 8) +
    theme_linedraw() +
    theme(text = element_text(size = 30),
          axis.title.x = element_blank(),
          axis.text.x = element_text(angle = 60, hjust = 1, vjust = 1),
          panel.grid = element_blank(),
          legend.position = "top",
    ) +
    scale_x_continuous(breaks = seq(0, 1, by = 0.25), limits = c(-0.2, 1)) +
    guides(color = "none", size = "none", 
           fill = guide_legend(override.aes = list(label = 1 : max_rank),
                               label.theme = element_blank(),
                               nrow = 1)) +
    scale_fill_manual(values = pal, labels = 1 : max_rank, name = "ranking") +
    facet_grid(setup~method)
  return(p)
}
