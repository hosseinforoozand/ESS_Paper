rm(list = ls())
library(circlize)


df<- read.csv("Input_to_produce_figure6.csv")
mat = data.matrix(df)
colnames(mat) = paste0("S", 1:32)
rownames(mat) = paste0("S", 1:32)

chordDiagram(mat,directional = 1, direction.type = c("diffHeight", "arrows"),
             link.arr.type = "big.arrow",diffHeight = -uh(4, "mm"), annotationTrack = c("name", "grid"),
             annotationTrackHeight = c(0.02, 0.04))





