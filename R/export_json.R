# Helper function to serialize savable Rforestry model to JSON
serialize_rforestry <- function(forest) {
  num_feature <- length(forest@processed_dta$processed_x)
  num_tree <- length(forest@R_forest)
  col_sds <- forest@colSd
  col_means <- forest@colMeans
  na_direction <-
    !is.null(attr(forest, "naDirection")) && forest@naDirection
  
  serialized_trees <- vector(mode = "list", length = num_tree)
  for (tree_id in 1:num_tree) {
    serialized_tree <-
      serialize_tree(forest@R_forest[[tree_id]], col_sds, col_means, na_direction)
    serialized_trees[[tree_id]] <- serialized_tree
  }
  
  serialized_forest <-
    list("n" = num_feature, "t" = serialized_trees)
  forest_json <-
    jsonlite::toJSON(serialized_forest, auto_unbox = TRUE, digits = 16)
  return(forest_json)
}


serialize_tree <- function(tree, col_sds, col_means, na_direction) {
  split_feat <- tree$var_id
  split_val <- tree$split_val
  node_values <- tree$weights
  na_left_count <- tree$naLeftCount
  na_right_count <- tree$naRightCount
  if (na_direction) {
    na_default_directions <- tree$naDefaultDirections
  }
  
  node_id <- 1
  node_value_i <- 1
  split_feat_i <- 2
  split_val_i <- 2
  na_count_i <- 1
  
  root_is_leaf <- split_feat[1] < 0
  if (root_is_leaf) {
    root_node = list("id" = 0, "v" = node_values[node_value_i] * col_sds[length(col_sds)] + col_means[length(col_means)])
    split_feat_i <- split_feat_i + 2
  } else {
    root_node = list(
      # Node ID
      "i" = 0,
      # Feature ID
      "f" = split_feat[1] - 1,
      # Splitting threshold
      "t" = split_val[1] * col_sds[split_feat[1]] + col_means[split_feat[1]],
      # Left node ID
      "l" = FALSE,
      # Right node ID
      "r" = FALSE
    )
  }
  serialized_tree <- list(root_node)
  
  prev_is_leaf <- FALSE
  incomplete_nodes <- c(0)
  while (split_feat_i <= length(split_feat)) {
    if (prev_is_leaf) {
      num_incomplete <- length(incomplete_nodes)
      prev_incomplete_node <- incomplete_nodes[num_incomplete]
      serialized_tree[[prev_incomplete_node + 1]]["r"] <- node_id
      incomplete_nodes <- incomplete_nodes[-num_incomplete]
    } else {
      serialized_tree[[node_id]]["l"] <- node_id
    }
    
    # If split node:
    if (split_feat[split_feat_i] > 0) {
      threshold <-
        split_val[split_val_i] * col_sds[split_feat[split_feat_i]] + col_means[split_feat[split_feat_i]]
      
      if (na_direction) {
        default_left = na_default_directions[na_count_i] == -1
      } else {
        default_left = na_left_count[na_count_i] >= na_right_count[na_count_i]
      }
      
      curr_node = list(
        # Node ID
        "i" = node_id,
        # Feature ID
        "f" = split_feat[split_feat_i] - 1,
        "t" = threshold,
        # Whether to default left for missing values
        'd' = default_left,
        # left node ID
        "l" = FALSE,
        # Right node ID
        "r" = FALSE
      )
      incomplete_nodes <- c(incomplete_nodes, node_id)
      split_feat_i <- split_feat_i + 1
      na_count_i + 1
      prev_is_leaf <- FALSE
    }
    # If leaf node:
    else {
      curr_node = list("i" = node_id,
                       # leaf value
                       "v" = node_values[node_value_i] * col_sds[length(col_sds)] + col_means[length(col_means)])
      node_value_i <- node_value_i + 1
      split_feat_i <- split_feat_i + 2
      prev_is_leaf = TRUE
    }
    serialized_tree <- c(serialized_tree, list(curr_node))
    split_val_i <- split_val_i + 1
    node_id <- node_id + 1
  }
  return(serialized_tree)
}