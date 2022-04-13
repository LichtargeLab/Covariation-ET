# Chen Wang @ 03/23/2022
# This code is a speed up version of CovET. It takes the Non-Gapped MSA  and 
# tree files generated from PyET package, then produce CovET output.
# It runs roughly 5 times faster than the original CovET code, and doesn't 
# generate large intermediate files.

# Use Rscript CovET_Rcpp_cmd.R <tree file> <msa file> <query id> <pair output file> \
# <single output file> <legacy output file> <cores>
# cores are optional, if not specified, 1 core will be used

cpp.function.path <- "CovET_func.cpp"

args <- commandArgs(trailingOnly = TRUE)

tree.path <- args[1]
msa.path <- args[2]
query.id <- args[3]
pair.output.path <- args[4]
single.output.path <- args[5]
legacy.output.path <- args[6]

if (length(args) == 7) {
  cores <- as.numeric(args[7])
} else {
  cores <- 1
}

suppressWarnings(suppressMessages(library(dplyr)))
suppressWarnings(library(tidyr))
suppressWarnings(library(stringr))
suppressWarnings(library(readr))
suppressWarnings(library(purrr))
suppressWarnings(library(ape))
library(parallel)
suppressWarnings(library(pbmcapply))
suppressWarnings(library(Rcpp))
options(dplyr.summarise.inform = FALSE)


############################################################################
# Processing tree
############################################################################
et.tree <- ape::read.tree(tree.path)

# The tree file produced by PyET sometimes have extra quote (') on the entry
# name. Need to remove the quotes before mapping.
tip.label.temp <- et.tree$tip.label
tip.label.temp[str_starts(tip.label.temp, "'")] <- str_sub(tip.label.temp[str_starts(tip.label.temp, "'")],
                                                           2, -1)
tip.label.temp[str_ends(tip.label.temp, "'")] <- str_sub(tip.label.temp[str_ends(tip.label.temp, "'")],
                                                         1, -2)
et.tree$tip.label <- tip.label.temp
rm(tip.label.temp)

tip.count <- length(et.tree$tip.label)
# Get root node number. Nodes are numbered after the tips. Root is the first
# node.
root.node.num <- tip.count + 1

# Get distance to root for each inner nodes, then identify the number of groups
# present at each node. 
# pyET.node.num is parsed from tree file generated from PyET output. Some nodes have
# same distances to root. The order for those nodes are arbitrary. In order to reproduce
# same output as PyET, the node order from PyET tree is directly used.
dist.to.root <- dist.nodes(et.tree)[,root.node.num]
dist.to.root <- tibble(node.num = names(dist.to.root),
                       dist = dist.to.root) %>%
  mutate(node.num = as.numeric(node.num)) %>%
  filter(node.num > tip.count) %>%
  mutate(node.name = et.tree$node.label[node.num-tip.count]) %>%
  mutate(pyET.node.num = str_sub(node.name, 6, -1),
         pyET.node.num = as.numeric(pyET.node.num)) %>%
  # in rare cases two nodes have the same distance to 
  # root, thus pick the one with smaller node.num to be 
  # closer to the root.
  arrange(pyET.node.num) %>%
  mutate(total.group = rank(pyET.node.num, ties.method = "first") + 1)

# Get most recent common ancestor between pair of sequences (tips) in the tree
mrca_mat <- mrca(et.tree)
diag(mrca_mat) <- 0
mrca_mat[lower.tri(mrca_mat)] <- 0

# This function breaks a sorted integer vector into start and end points.
# The values in the vector are end points of the break down segment.
BreakGroups <- function(x) {
  start.vec <- (x + 1) [1:(length(x)-1)]
  end.vec <- x[2:length(x)]
  output <- tibble(start = start.vec, end = end.vec)
  return(output)
}

# We first identify the two groups of sequences (tips) at each inner node.
# The the unique sequences groups are identified and the scale for each group 
# that contributes to the final CovET score are calculated. The start and end points 
# (index) for each group are also identified. Here each tip is represented as a 
# number. The order the same as how they appeared in the tree. Neighboring tips
# will have will be labeled with N and N+1.
# Then for each pair we just need to calculate the trace for each group in the 
# processed.tree file, and combine them to get the final CovET score.
processed.tree <- dist.to.root %>%
  mutate(split.point = map_dbl(node.num, ~{which(mrca_mat == .)[1] %/% tip.count})) %>%
  mutate(accumulated.points = accumulate(split.point, c)) %>%
  mutate(accumulated.points = map(accumulated.points, ~sort(c(., 0, tip.count)))) %>%
  add_row(node.num = 0, dist = -1, total.group = 1, 
          accumulated.points = list(c(0, tip.count)),
          .before = 1) %>%
  mutate(groups.at.node = map(accumulated.points, BreakGroups)) %>%
  select(node.num, total.group, groups.at.node) %>%
  unnest(cols = groups.at.node) %>%
  mutate(scale = 1/total.group) %>%
  group_by(start, end) %>%
  summarize(scale = sum(scale)) %>%
  mutate(tip.n = end - start + 1,
         group.size = tip.n * (tip.n - 1)/2) 

# There is a fixed penalty for each group that has only 1 seq.
fixed.penalty <- sum(processed.tree$scale[processed.tree$tip.n==1])
processed.tree <- processed.tree[processed.tree$tip.n>1,]

############################################################################
# Processing MSA
############################################################################
# Load MSA and trim name till space. The trimming is because the tree
# file from PyET trims after space.
msa <- read.FASTA(msa.path, type = "AA") %>%
  as.character()
trim.name <- str_split_fixed(names(msa), " ", n = 2)[,1]
names(msa) <- trim.name

# Reorder sequences in msa to match the order in the tree, so the start/end in the
# processed.tree file can be used to index sequences in the MSA.
msa <- msa[et.tree$tip.label] %>%
  reduce(rbind)
dimnames(msa) <- NULL

if (nrow(msa) != length(et.tree$tip.label)) {
  cat("Not all entries in the MSA are mapped to ET tree.\nCheck entry name.\n")
  msa <- NULL
}

# Convert AA to double digits
num.to.AA <- 1:24 + 10
names(num.to.AA) <- c("-","A","C","D","E",
                      "F","G","H","I","K",
                      "L","M","N","P","Q",
                      "R","S","T","V","W","Y",
                      "X","V","Z")

# msa.num is the numerical representation of the MSA.
msa.num <- num.to.AA[msa] %>%
  matrix(., nrow = nrow(msa))

# This function produces two N by N matrices for each position in the MSA file, where
# N is the number of sequences in the MSA.
# The logi matrix stores if seq_m, seq_n at position i are the same AAs
# or not.
# The vari matrix stores the four digit representation of seq_m, seq_n 
# at position i.
GeneratePositionMatrix <- function(msa.num, pos.i) {
  pos.i.seq <- msa.num[,pos.i]
  tip.n <- length(pos.i.seq)
  mat.1 <- matrix(pos.i.seq, ncol = tip.n, nrow = tip.n, byrow = TRUE) 
  mat.2 <- matrix(pos.i.seq, ncol = tip.n, nrow = tip.n, byrow = FALSE) 
  
  mat.logi <- mat.1 == mat.2
  mat.logi[lower.tri(mat.logi)] <- TRUE
  diag(mat.logi) <- TRUE
  
  mat.vari <- mat.1 + 100*mat.2
  mat.vari[lower.tri(mat.vari)] <- 0
  diag(mat.vari) <- 0
  return(list(mat.logi, mat.vari))
}

# A list that stores all the matrices.
# the logi matrix for position i is at 2*i-1
# the vari matrix for position i is at 2*i
pos.mat <- mapply(GeneratePositionMatrix, 
                  pos.i = 1:ncol(msa), MoreArgs = list(msa.num = msa.num))

############################################################################
# Calculate CovET scores and produce CovET pair output
############################################################################
# Load c++ functions that calculate the trace for a group
# These functions are written in cpp for faster processing
sourceCpp(cpp.function.path)

# Funciton to trace a given pair in the MSA.
TracePair <- function(pos.mat, processed.tree, fixed.penalty, pos.i, pos.j) {
  # A non concerted pair has to have one position varies the other pair does not vari
  non.concert.rev.index <- ((pos.mat[[2*pos.i-1]] + pos.mat[[2*pos.j-1]]) != 1)
  # Pair variations are represented as a 8 digit integer.
  non.concert.mat <- (pos.mat[[2*pos.i]] + 10000*pos.mat[[2*pos.j]])
  non.concert.mat[non.concert.rev.index] <- 0
  group.trace <- TraceGroupC_vec(processed.tree$start,
                                 processed.tree$end,
                                 processed.tree$group.size,
                                 non.concert.mat)
  output <- sum(group.trace*processed.tree$scale) + fixed.penalty + 1
  return(output)
}

pair.output <- crossing(Position_i = 1:ncol(msa),
                        Position_j = 1:ncol(msa)) %>%
  filter(Position_i < Position_j) 

CovET.trace <- pbmcapply::pbmcmapply(TracePair, pos.i = pair.output$Position_i,
                                     pos.j = pair.output$Position_j,
                                     MoreArgs = list(pos.mat = pos.mat, processed.tree = processed.tree,
                                                     fixed.penalty = fixed.penalty), 
                                     mc.cores = cores,
                                     ignore.interactive = TRUE)

# This function computes the Variability_Count for a given position pair in the
# alignment.
GetVariCount <- function(pos.i, pos.j, msa.num) {
  output <- unique(msa.num[,pos.i]+msa.num[,pos.j]*100) %>% 
    length()
  return(output)
}

pair.output$Score <- CovET.trace
pair.output$Variability_Count <- mapply(GetVariCount, pos.i = pair.output$Position_i, 
                                        pos.j = pair.output$Position_j,
                                        MoreArgs = list(msa.num = msa.num))

query.seq <- msa[which(et.tree$tip.label == query.id),] 
pair.output <- pair.output %>%
  mutate(Rank = dense_rank(Score)) %>%
  mutate(Coverage = rank(Score, ties.method = "max")/n()) %>%
  mutate(Score = round(Score, digits = 3),
         Coverage = round(Coverage, digits = 3)) %>%
  mutate(Query_i = query.seq[Position_i],
         Query_j = query.seq[Position_j]) %>%
  select(ends_with("_i"), ends_with("_j"), Variability_Count, Rank, Score, Coverage)


############################################################################
# Convert pair output to single residue output and produce legacy ET file
############################################################################

CovET_Pair_to_Single <- function(df) {
  output <- df %>%
    dplyr::rename(Position_i = Position_j, Position_j = Position_i, Query_i = Query_j, Query_j = Query_i) %>%
    bind_rows(df, .) %>%
    group_by(Position_i, Query_i) %>%
    filter(Rank == min(Rank)) %>%
    summarize(min.pair.rank = min(Rank),
              Score = min(Score)) %>%
    ungroup() %>%
    mutate(Rank = dense_rank(min.pair.rank),
           Coverage = rank(min.pair.rank, ties.method = "max")/n()) %>%
    mutate(Coverage = round(Coverage, digits = 3)) %>%
    select(Position = Position_i, Query = Query_i, Rank, Score, Coverage)
  return(output)
}

single.vari <- apply(msa, 2, FUN = unique)

single.output <- CovET_Pair_to_Single(pair.output) %>%
  mutate(Variability_Characters = sapply(single.vari, paste0, collapse = ""),
         Variability_Count = sapply(single.vari, length)) %>%
  select(Position, Query, Variability_Count, Variability_Characters,
         Rank, Score, Coverage)

# Reformat single.output to legacy ET file 
ReformatET <- function(df) {
  output <- df %>%
    select(align = Position, resi = Position, type = Query, rank = Rank, vari = Variability_Characters,
           rho = Score, coverage = Coverage) %>%
    mutate(vari = str_remove_all(vari, ",")) %>%
    as.data.frame()
  return(output)
}

write_ET <- function(df, file) {
  write_file("% alignment#  residue#      type      rank              variability           rho     coverage\n",
             file)
  gdata::write.fwf(df, file, width = c(10,10,10,10,32,10,10), colnames = FALSE, justify="right", sep = "", append = TRUE)
}

# Write output

write_tsv(pair.output, pair.output.path)
write_tsv(single.output, single.output.path)
single.output %>% 
  ReformatET() %>%
  write_ET(file = legacy.output.path)

