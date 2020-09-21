rm(list = ls())
library(circlize)


df<- read.csv("Input_to_produce_figure8.csv")

border_chosen = data.frame(c("S15","S15","S15","S15","S15","S15","S15","S15","S15","S15","S15","S15","S15","S15","S15","S15"),
                           c("S1","S14","S28","S6","S30","S16","S13","S5","S10","S32","S27","S12","S29","S4","S11","S7"), 
                           c(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1))



chordDiagram(df,directional = 1, direction.type = c("diffHeight", "arrows"),link.lwd = 1, 
             link.border = border_chosen,
             link.arr.type = "big.arrow",diffHeight = -uh(4, "mm"), annotationTrack = c("name", "grid"),
             annotationTrackHeight = c(0.02, 0.04))

