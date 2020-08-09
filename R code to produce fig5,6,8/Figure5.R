# Reference four-set diagram
library(VennDiagram)
venn.plot <- draw.quad.venn(
  area1 = 88,
  area2 = 80,
  area3 = 50,
  area4 = 50,
  n12 = 44,
  n13 = 27,
  n14 = 32,
  n23 = 38,
  n24 = 32,
  n34 = 20,
  n123 = 18,
  n124 = 17,
  n134 = 11,
  n234 = 13,
  n1234 = 6,
  category = c("A", "D", "B", "C"),
  fill = c("orange", "red", "green", "blue"),
  lty = "dashed",
  cex = 2,
  cat.cex = 2,
  cat.col = c("orange", "red", "green", "blue")
);


