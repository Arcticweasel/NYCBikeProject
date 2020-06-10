# R-Script for helper functions mainly used in pre-processing data for Artificial Neural Networks (ANN)
# Include this script into your project with the command: source("<path>/ANNUtils.r")

# Transformation functions
as.numeric.factor <- function(x) {as.numeric(levels(x))[x]}

# Winsorize

winsorize <- function(x, fraction = .05) {
  if(length(fraction) != 1 || fraction < 0 || fraction > 0.5) {
    stop("bad value for 'fraction'")
  }
  lim <- quantile(x, probs=c(fraction, 1-fraction))
  x[x < lim[1]] <- lim[1]
  x[x > lim[2]] <- lim[2]
  return(x)
}

# Dummification of categorial (nominal or ordinal) variables
# Instead of using default 0/1 pairs, effectcoding allows to set -1/1

dummify <- function(x, colname, sep = "_", with_sort = F, effectcoding = FALSE) {
  f <- as.factor(x)
  if (with_sort) { l <- sort(levels(f)) } else { l <- levels(f)}
  n <- nlevels(f)
  du <- data.frame(x)
  colnames(du)[1] <- c(colname)
  if (n == 1) {
    s <- paste(colname, sprintf("%d",1), sep = sep) 
    du[s] <- 1
  }
  else {
    if (!effectcoding) {zerovalue <- 0} else {zerovalue <- -1}
    for (i in 1:(n-1)) {
      s <- paste(colname, sprintf("%s",l[i]), sep = sep) 
      du[s] <- ifelse(x == l[i], 1, zerovalue)
    }
  }
  return(du)
}

# Effectcoding
# If a variable consist of 0/1 pairs, effectcoding transforms the values to -1/1 pairs

effectcoding <- function(x) {
  return(ifelse(x == 0, -1, 1))
}

# Builds a one-hot-vector
# A categorial outcome (y), a nominal or ordinal variable, must be rebuild to a so called "one hot vector";
# Within a one-hot-vector each level of a factor is rebuild in the form (0|1,0|1,0|1,...)

to_one_hot <- function(x) {
  f <- as.factor(x)
  n <- nlevels(f)
  m <- matrix(0, nrow = NROW(x), ncol = n)
  colnames(m) <- levels(f)
  for (i in 1:NROW(x)){
    m[i,f[[i]]] <- 1
  }
  return(m)
}

# Builds a lagged data set

lags <- function(x, k = 1, filled = NA, recursive = F) {
  N <- NROW(x)
  if (!recursive) {
    if (k > 0) {
      return(c(rep(filled,k),x)[1:N])
    }
    else {
      return(c(x[(-k+1):N],rep(filled,-k)))
    }
  }
  else {
    l <- list()
    if (k > 0) {
      for (i in 1:k) {
        l[[i]] <- c(rep(filled,i),x)[1:N]
      }
    }
    else {
      for (i in 1:abs(k)) {
        l[[i]] <- c(x[(i+1):N],rep(filled,i))
      }
    }
    return(do.call(cbind, l))
  }
}

# Builds a stationary data/time series with optional lags

build_stationary <- function(x, k = 0, filled = NA) {
  d <- diff(x)
  if (k > 0) {
    l <- lags(d, k, filled, recursive = T)
    return(cbind(d, l))
  }
  else {
    return(d)
  }
}

# Inverts a difference data series

invert_differencing <- function(differences, origin) {
  lp <- length(differences)
  lo <- length(origin)
  prediction <- NA
  # Only a start value is given for invert differencing
  # The result of the first invert differencing is basis for second invert differencing etc.
  if (lo == 1) {
    prediction <- numeric(lp)
    prediction <- diffinv(differences, xi = origin) #cumsum(c(origin,differences))
    prediction <- prediction[-1]
  } 
  # The original series is iteratively be used for invert differencing
  else {
    if (lo != lp) stop("length of predictions and origins are not equal")
    prediction <- numeric(lp)
    prediction <- sapply(1:lo, function(x)
      {prediction[x] <- differences[x] + origin[x]})
  }
  return(prediction)
}

# Builds a lagged data/time series without differentation
## y, lag1, lag2, lag3 etc.

build_lagged_timeseries <- function(tseries, lag = 1, with_reorder = T) {
  overall_timesteps <- 0
  df <- as.data.frame(t(sapply(1:(length(tseries) - lag + overall_timesteps), function(x) 
    tseries[x:(x + lag - overall_timesteps)])))
  if (with_reorder) {df <- df[,c(ncol(df):1)]}
  return(df)
}

# Min-Max-Normalization of data
# Note: Test data must be normalized with train data scales (min,max)

normalize <- function(x, minx = NULL, maxx = NULL) {
  if (is.null(minx) && is.null(maxx)) {
    return ((x - min(x))/(max(x)-min(x)))
  } else {
    return ((x - minx)/(maxx-minx))
  }
}

denormalize <- function(x, minx, maxx) {
  x*(maxx-minx) + minx
}

# Outputs a named list
## min  : Vector of minima of columns
## max  : Vector of maxima of columns
## train: Normalized train data set
## test : Normalized test data set

normalize_data <- function(traindata, testdata) {
  l <- list()
  l[[1]] <- sapply(traindata, min)
  l[[2]] <- sapply(traindata, max)
  l[[3]] <- as.data.frame(sapply(traindata, normalize))
  l[[4]] <- as.data.frame(mapply(normalize, testdata, l[[1]], l[[2]]))
  names(l) <- c("min","max","train","test")
  return(l)
}

# Resampling of imbalanced data
## Oversampling: copy n rows of minority class (under-represented category)
## Undersamping: delete n% rows of majority class (over-represented category)
## Synthetic Minority Oversampling Technique (SMOTE): create n synthetic rows of minority class of k nearest neighbors 
## http://rikunert.com/SMOTE_explained

resample.imbalanced <- function(dataset, target, n = 1, k = 1, type = "smote") {
  type_names <- c("oversampling","undersampling","smote")
  df <- as.data.frame(dataset)
  cnames <- colnames(df)
  X <- df[,-target] # Extract feature matrix
  y <- df[,target] # Extract target vector
  n_target <- table(y) # Number of instances of each class
  # Oversampling
  if (type == type_names[1]) {
    min_class <- names(which.min(n_target)) # Name of minority class
    X_min_all <- subset(df, y == min_class) # under-represented categories
    df <- rbind(df, do.call(rbind, replicate(n, X_min_all, simplify = F)))
  } else {
  # Undersampling
  if (type == type_names[2]) {
    max_class <- names(which.max(n_target)) # Name of majority class
    N <- nrow(df[y == max_class,]) # number of over-represented categories
    n_ <- round(N*n, digits = 0)
    df <- df[-c(sort(sample(which(y == max_class), n_, replace = F), decreasing = F)),]
  } else {
  # SMOTE
  if (type == type_names[3]) {
  }    
    min_class <- names(which.min(n_target)) # Name of minority class
    X_min_all <- subset(X, y == min_class)[sample(min(n_target)),] # all minority feature values in shuffled order
    x1 <- X_min_all[1,] # reference sample with feature values
    X_min <- X_min_all[-1,] # remaining minority samples with feature values
    
    distances <- apply(X_min, 1, euclidean_distance, x2 = x1) # euclidean distances from reference sample to all other samples
    dist_inst <- data.frame(index=c(1:NROW(distances)), ed=distances) # euclidean distances and row indices
    dist_inst <- dist_inst[order(dist_inst$ed),] # ascending ordering
    idx <- dist_inst$index[(1:k)] # indices of k nearest neighbors
    X_nearest_neighbors <- X_min[idx,] # k nearest neighbors
    
    fl <- list()
    for (i in 1:n) {
      x2 <- X_nearest_neighbors[sample(NROW(X_nearest_neighbors),1),] # random remaining sample of feature values
      v <- c()
      for (j in 1:length(x1)) {
        v1 <- as.numeric(x1[j]) # feature value boundary 1 of minority class
        v2 <- as.numeric(x2[j]) # feature value boundary 2 of minority class
        random_value <- sample(v1:v2,1)
        v <- c(v,random_value)
      }
      fl[[i]] <- v
    }
    new_feature_values <- as.data.frame(do.call(rbind, fl))
    colnames(new_feature_values) <- names(X)
    # cn <- colnames(new_feature_values)
    # get_index <- function(x) {return(which(colnames(df) == x))}
    # idx <- unlist(lapply(as.list(cn), get_index)) # column indices of features
    # l[[1]] <- min_class
    # l[[2]] <- as.data.frame(new_feature_values)
    # names(l) <- c("minority_class","feature_values")
    # return(l)
    
    cbind.columns <- function(dataset, new_column, after) {
      if (after == 0) {
        return(cbind.data.frame(new_column,dataset))
      } else {
        if (after >= NCOL(dataset)) {
          return(cbind.data.frame(dataset,new_column))
        } else {
          return(cbind.data.frame(d[,1:(after),drop=F], y, d[,(after+1):length(d),drop=F]))
        } 
      }
    }
    new_subset <- (cbind.columns(new_feature_values, rep(min_class, NROW(new_feature_values)), (target-1)))
    colnames(new_subset) <- cnames
    df <- rbind(df, new_subset)
  }}
  return(df)
}

# Get ANN lags from ARIMA(X) lags

as.lags <- function(arima_lags = 0, type = "univariate") {
  type_names <- c("univariate","multivariate")
  if (type == type_names[2]) {
    l <- arima_lags
  } else {
    l <- ifelse(arima_lags < 1, 1, arima_lags)
  }
  return(l)
}

# Get timesteps from lags
## Univariate: timesteps = lags
## Multivariate: timesteps = lags + 1

as.timesteps <- function(lags = 1, type = "univariate") {
  type_names <- c("univariate","multivariate")
  tsteps <- lags
  if (type == type_names[2]) {tsteps <- lags + 1}
  return(tsteps)
}

# Resampling time series data
# Within a univariate time series, y(t) is explained by past y(t-1), y(t-2) etc. Therefore the last record of the
# feature set must be deleted, because there is no Y-value for it. Resampling of the Y-values must start at timesteps + 1.
# That is different for a multivariate time series. For y(t), the corresponding features at time t are already given.
# Resampling must start at timesteps. In our case, resampling takes care to build a quasi-multivariate time series.
# Note: Our starting point is a univariate time series. From that, we extract an implicit x as lag 1. The x is a lagged value!
#       That's the reason why we must treat the time series as a univariate time series although we have a x.
#       But this x isn't an explicit x. Every non-lagged x is an explicit x and spawn a multivariate time series.

resample.X <- function(X, timesteps, sep = "_") {
  df <- as.data.frame(X)
  cnames <- colnames(df)
  s <- do.call(paste, list(cnames, "%s%d", sep = ""))
  newnames <- c()
  for (i in 1:timesteps) {
    parts <- do.call(sprintf, list(fmt = s, sep, i))
    newnames <- c(newnames, parts)
  }
  newcol <- ncol(df) * timesteps
  n <- nrow(df) - timesteps + 1
  dflist <- list()
  for (i in 1:n) {
    dflist[[i]] <- matrix(t(df[(i+timesteps-1):i,]), nrow = 1)
  }
  res <- do.call(rbind, dflist)  
  colnames(res) <- newnames
  return(res)
}

resample.y <- function(y, timesteps) {
  y <- as.vector(y)
  return(y[timesteps:NROW(y)]) #return(y[seq(timesteps, length(y), 1)])
}

# K-fold cross validation
# Splits a data set in partial sets, so called folds, and creates a list of folds
# For time series data the argument random must set equal to FALSE because the order of the records can't be changed

cross_validation_split <- function(dataset, folds = 3, foldname = "fold", random = FALSE) {
  df <- as.data.frame(dataset)
  if (random) {df <- df[sample(1:NROW(df)),]} else {df <- df[1:NROW(df),]}
  fold_size <- as.integer(NROW(df)/folds)
  fold_list <- list()
  listnames <- c()
  for (i in 1:folds) {
    fold_list[[i]] <- head(df, n = fold_size)
    df <- tail(df, -fold_size)
    listnames <- c(listnames, sprintf(fmt = paste(foldname, "%d", sep = ""), i))
  }
  names(fold_list) <- listnames  
  return(fold_list)
}

# Required libraries
require(keras)

# Data format for SLP/MLP
# X: Features must be in a matrix
# Y: Outcomes must be in a matrix

as.MLP.X <- function(X){
  X.tensor <- as.matrix(X)
  return(X.tensor)
}

as.MLP.Y <- function(y){
  ifelse (is.factor(y), Y.tensor <- as.integer(y)-1, Y.tensor <- y)
  Y.tensor <- as.matrix(Y.tensor)
  return(Y.tensor)
}

# Build SLP/MLP architecture
build_mlp <- function(features, hidden = NULL, dropout = NULL, output = c(1,"linear"), loss, optimizer, metrics) {
  mlp_model <- keras_model_sequential()
  # SLP
  if (is.null(hidden)) {
    mlp_model %>% layer_dense(units = output[1], activation = output[2], input_shape = features)
  }
  # MLP
  else {
    h <- as.data.frame(hidden)
    N <- NROW(h)
    # First hidden layer
    mlp_model %>% layer_dense(units = h[1,1], activation = h[1,2], input_shape = features)
    d <- 1 # dropout layers to prevent overfitting
    D <- ifelse(!(is.null(dropout)),NROW(dropout),0)
    if (D > 0) {mlp_model %>% layer_dropout(rate = dropout[d]); d <- d + 1}
    # Further hidden layers
    i <- 1 # hidden layers
    while (i < N) {
      mlp_model %>% layer_dense(units = h[i+1,1], activation = h[i+1,2])
      i <- i + 1
      if (d <= D) {mlp_model %>% layer_dropout(rate = dropout[d]); d <- d + 1}
    }
    # Output layer
    mlp_model %>% layer_dense(units = output[1], activation = output[2])
  }
  mlp_model %>% compile(loss = loss, optimizer = optimizer, metrics = metrics)
  return(mlp_model)
}

# Fit SLP/MLP model
fit_mlp <- function(X, y, epochs = 100, batchsize = 1, validation_split = 0.2,
                    k.fold = NULL, k.optimizer = NULL,
                    hidden = NULL, dropout = NULL, output_activation = "linear", loss, optimizer, metrics) {
  l <- list() # result
  l_names <- c("hyperparameter","model","avg_qual")
  l_hyperparameter_names <- c("features","output_units")
  
  # SLP/MLP data format
  x.train <- as.MLP.X(X)
  y.train <- as.MLP.Y(y)
  
  # Calculated Hyperparameters
  ann.features     <- NCOL(x.train)
  ann.output_units <- NCOL(y.train)
  l[[1]] <- list(ann.features, ann.output_units)
  names(l[[1]]) <- l_hyperparameter_names
  
  # Use "<<-" for usage in global environment
  build_mlp_model <- function() {
    mlp.model <- build_mlp(features = ann.features,
                           hidden = hidden,
                           dropout = dropout,
                           output = c(ann.output_units, output_activation),
                           loss = loss,
                           optimizer = optimizer,
                           metrics = metrics)
  }

  if (is.null(k.fold)) {
    # Build model
    l[[2]] <- build_mlp_model()
    # Train/Fit the model
    l[[2]] %>% fit(x.train, y.train, epochs = epochs, batch_size = batchsize, validation_split = validation_split)
    # Named list
    names(l) <- l_names[1:2]
  } 
  else {
    k <- k.fold
    # List of data sets folds
    x.fold_datasets <- cross_validation_split(X, k)
    y.fold_datasets <- cross_validation_split(y, k)
    
    # Quality measure(s)
    all_qual_histories <- NULL
    all_scores <- c()
    
    # Folds loop
    for (i in 1:(k-1)) {
      # Extract training and validation fold
      x.train.fold <- as.MLP.X(x.fold_datasets[[i]])
      y.train.fold <- as.MLP.Y(y.fold_datasets[[i]])
      x.val.fold <- as.MLP.X(x.fold_datasets[[i+1]])
      y.val.fold <- as.MLP.Y(y.fold_datasets[[i+1]])
      
      # Build model
      l[[2]] <- build_mlp_model()
      
      # Train/fit model
      history <- l[[2]] %>%
        fit(x = x.train.fold, y = y.train.fold,
            epochs = epochs, batch_size = batchsize,
            validation_data = list(x.val.fold, y.val.fold))
      
      # Store training results
      results <- l[[2]] %>% evaluate(x.val.fold, y.val.fold, verbose = 0)
      m <- l[[2]]$metrics_names[2]
      all_scores <- c(all_scores, results$m) #$mean_absolute_error
      qual_history <- history$metrics[[4]] #$val_mean_absolute_error
      all_qual_histories <- rbind(all_qual_histories, qual_history)
    }
    
    # Build up history of successively mean k-fold Validation scores
    average_qual_history <- data.frame(
      epoch = seq(1: ncol(all_qual_histories)),
      validation_qual = apply(all_qual_histories, 2, mean)
    )

    l[[3]] <- average_qual_history
    names(l) <- l_names
    
    # Train/Fit the final or generalized model
    # The function can deal with min or max optimizations
    if (!(is.null(k.optimizer))) {
      if (k.optimizer == "min") {
        opt_epoch <- average_qual_history$epoch[which.min(average_qual_history$validation_qual)]
      } else {
        opt_epoch <- average_qual_history$epoch[which.max(average_qual_history$validation_qual)]
      }
      l[[2]] <- build_mlp_model()
      l[[2]] %>% fit(x.train, y.train, epochs = opt_epoch, batch_size = batchsize, validation_split = validation_split)
    }
  } 
  return(l)
}

# Predict with SLP/MLP
predict_mlp <- function(mlp, X) {
  # SLP/MLP data format
  x.test <- as.MLP.X(X)

  # Prediction
  y.predict <- mlp %>% predict(x.test)
  
  return(y.predict)
}

# Data format for LSTM
# X: Features must be in a 3D-array with following dimensionens
##   Samples  : Number of records
##   Timesteps: Number of different periods within a record (sample)
##   Features : Number of features (x) within a sequence of period
# Y: Outcomes must be in a 2D-array with the dimensions Samples and Units (number of output units)

as.LSTM.X <- function(X, timesteps){
  X.tensor <- as.matrix(X) #data.matrix()
  features <- as.integer(NCOL(X.tensor)/timesteps) # Number of features for timeseries
  X.tensor <- array(data = X.tensor, dim = c(NROW(X.tensor), timesteps, features))
  return(X.tensor)
}

as.LSTM.Y <- function(y){
  ifelse (is.factor(y), Y.tensor <- as.integer(y)-1, Y.tensor <- y)
  Y.tensor <- as.matrix(Y.tensor)
  output_units <- as.integer(NCOL(Y.tensor)) # Number of output units
  Y.tensor <- array(data = Y.tensor, dim = c(NROW(Y.tensor), output_units))
  return(Y.tensor)
}

# Build LSTM architecture
# Univariate time series  : usually stateful = T and batchsize = 1; return_sequences = F
# Multivariate time series: usually stateful = F and batchsize = NULL; return_sequences = T
build_lstm <- function(features, timesteps = 1, batchsize = NA, hidden, dropout = NULL, output = c(1,"linear"), 
                       stateful = FALSE, return_sequences = TRUE,
                       loss, optimizer, metrics) {
  lstm_model <- keras_model_sequential()
  h <- as.data.frame(hidden)
  N <- NROW(h)
  rs <- return_sequences
  if (N == 1) rs <- F
  # First hidden layer
  if (is.na(batchsize)) {
    lstm_model %>% layer_lstm(units = h[1,1], input_shape = c(timesteps, features), activation = h[1,2], stateful = stateful, return_sequences = rs)
  } else {
    lstm_model %>% layer_lstm(units = h[1,1], batch_input_shape = c(batchsize, timesteps, features), activation = h[1,2], stateful = stateful, return_sequences = rs)
  }
  d <- 1 # dropout layers to prevent overfitting
  D <- ifelse(!(is.null(dropout)),NROW(dropout),0)
  if (D > 0) {lstm_model %>% layer_dropout(rate = dropout[d]); d <- d + 1}
  # Further hidden layers
  i <- 1
  while (i < N) {
    if ((i == (N-1)) && (rs == T)) {rs <- !rs}
    lstm_model %>% layer_lstm(units = h[i+1,1], activation = h[i+1,2], stateful = stateful, return_sequences = rs)
    i <- i + 1
    if (d <= D) {lstm_model %>% layer_dropout(rate = dropout[d]); d <- d + 1}
  }
  # Output layer
  lstm_model %>% layer_dense(units = output[1], activation = output[2])

  lstm_model %>% compile(loss = loss, optimizer = optimizer, metrics = metrics)
  return(lstm_model)
}

# Fit LSTM model
fit_lstm <- function(X, y, timesteps = 1, epochs = 100, batchsize = c(NA,1), validation_split = 0.2,
                     k.fold = NULL, k.optimizer = NULL,
                     hidden, dropout = NULL, output_activation = "linear", stateful = FALSE, return_sequences = TRUE,
                     loss, optimizer, metrics) {
  l <- list() # result
  l_names <- c("hyperparameter","model","avg_qual")
  l_hyperparameter_names <- c("features","output_units")
  
  # LSTM data format
  lstm.x.train <- as.LSTM.X(X, timesteps)
  lstm.y.train <- as.LSTM.Y(y)

  # Calculated Hyperparameters
  ann.features <- as.integer(NCOL(as.matrix(X))/timesteps) # Number of features
  ann.output_units <- as.integer(NCOL(as.matrix(y))) # Number of output units
  l[[1]] <- list(ann.features, ann.output_units)
  names(l[[1]]) <- l_hyperparameter_names
  
  # Use "<<-" for usage in global environment
  build_lstm_model <- function() {
    lstm.model <- build_lstm(features = ann.features,
                             timesteps = timesteps,
                             batchsize = batchsize[1],
                             hidden = hidden,
                             dropout = dropout,
                             output = c(ann.output_units, output_activation),
                             stateful = stateful,
                             return_sequences = return_sequences,
                             loss = loss,
                             optimizer = optimizer,
                             metrics = metrics)
  }
  
  if (is.null(k.fold)) {
    # Build and fit the model
    l[[2]] <- build_lstm_model()
    names(l) <- l_names[1:2]
    if (stateful == T) {
      for (i in 1:epochs) {
        l[[2]] %>% fit(lstm.x.train, lstm.y.train, epochs = 1, batch_size = batchsize[2], verbose = 1, shuffle = FALSE)
        l[[2]] %>% reset_states()
      }
    } else {
      l[[2]] %>% fit(lstm.x.train, lstm.y.train, epochs = epochs, batchsize = batchsize[2])
    }
  } 
  else {
    k <- k.fold
    # List of data sets folds
    x.fold_datasets <- cross_validation_split(X, k)
    y.fold_datasets <- cross_validation_split(y, k)
    
    # Quality measure(s)
    all_qual_histories <- NULL
    all_scores <- c()
    
    # Folds loop
    for (i in 1:(k-1)) {
      # Extract training and validation fold
      x.train.fold <- as.LSTM.X(x.fold_datasets[[i]], timesteps)
      y.train.fold <- as.LSTM.Y(y.fold_datasets[[i]])
      x.val.fold <- as.LSTM.X(x.fold_datasets[[i+1]], timesteps)
      y.val.fold <- as.LSTM.Y(y.fold_datasets[[i+1]])
      
      # Build model
      l[[2]] <- build_lstm_model()
      
      # Train/fit model
      history <- l[[2]] %>%
        fit(x = x.train.fold, y = y.train.fold,
            epochs = epochs, batch_size = batchsize[2],
            validation_data = list(x.val.fold, y.val.fold))
      
      # Store training results
      results <- l[[2]] %>% evaluate(x.val.fold, y.val.fold, verbose = 0)
      m <- l[[2]]$metrics_names[2]
      all_scores <- c(all_scores, results$m)
      qual_history <- history$metrics[[4]]
      all_qual_histories <- rbind(all_qual_histories, qual_history)
    }
    
    # Build up history of successively mean k-fold Validation scores
    average_qual_history <- data.frame(
      epoch = seq(1: ncol(all_qual_histories)),
      validation_qual = apply(all_qual_histories, 2, mean)
    )

    l[[3]] <- average_qual_history
    names(l) <- l_names
    
    # Train/Fit the final or generalized model
    # The function can deal with min or max optimizations
    if (!(is.null(k.optimizer))) {
      if (k.optimizer == "min") {
        opt_epoch <- average_qual_history$epoch[which.min(average_qual_history$validation_qual)]
      } else {
        opt_epoch <- average_qual_history$epoch[which.max(average_qual_history$validation_qual)]
      }
      l[[2]] <- build_lstm_model()
      l[[2]] %>% fit(lstm.x.train, lstm.y.train, epochs = opt_epoch, batch_size = batchsize[2], validation_split = validation_split)
    }
  } 
  return(l)
}

# Quality measures

## Mean Absolute Error
mae <- function(actual, predicted) {
  error <- actual - predicted
  return(mean(abs(error)))
}

## Mean Absolute Percentage Error
mape <- function(actual, predicted){
  error <- actual - predicted
  return(mean(abs(error/actual))*100)
}

## Root Mean Square Error
rmse <- function(actual, predicted) {
  error <- actual - predicted
  return(sqrt(mean(error^2)))
}

## Variance Coefficient
vc <- function(actual, predicted) {
  error <- actual - predicted
  return(sqrt(mean(error^2))/mean(actual))
}

# Statistical techniques

euclidean_distance <- function(x1, x2) {return(sqrt(sum((x1-x2)^2)))}

# Machine Learning algorithms

k_nearest_neighbors <- function(y, X, test, k = 1) {
  distances <- apply(X, 1, euclidean_distance, x2 = test) # calculate euclidean distances (ed)
  df <- data.frame(index=c(1:NROW(distances)), ed=distances) # build up data.frame with index and ed
  df <- df[order(df$ed),] # reorder data.frame in ascending order for ed
  idx <- df$index[(1:k)] # extract k minimum indices
  neighbors <- y[idx] # get k target classes/categories
  n_neighbors <- table(neighbors) # number of instances of each class
  majority_class <- names(which.max(n_neighbors)) # name of the majority class
  return(majority_class)
}
