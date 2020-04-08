# R-Script for helper functions mainly used in pre-processing data for Artificial Neural Networks (ANN)
# Include this script into your project with the command: source("<path>/ANNUtils.r")

# Transformation functions
as.numeric.factor <- function(x) {as.numeric(levels(x))[x]}

# Winsorize -> ausreißer anpassen

winsorize <- function (x, fraction = .05) {
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

dummify <- function (x, colname, sep = "_", with_sort = F, effectcoding = FALSE) {
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

# Min-Max-Normalization of data
# Note: Test data must be normalized with train data scales (min,max)

normalize <- function (x, minx = NULL, maxx = NULL) {
  if (is.null(minx) && is.null(maxx)) {
    return ((x - min(x))/(max(x)-min(x)))
  } else {
    return ((x - minx)/(maxx-minx))
  }
}

denormalize <- function(x, minx, maxx) {
  x*(maxx-minx) + minx
}

# Resampling time series data
# Within a univariate time series, y(t) is explained by past y(t-1), y(t-2) etc. Therefore the last record of the
# feature set must be deleted, because there is no Y-value for it. Resampling of the Y-values must start at timesteps + 1.
# That is different for a multivariate time series. For y(t), the corresponding features at time t are already given.
# Resampling must start at timesteps. In our case, resampling takes care to build a quasi-multivariate time series.
# Note: Our starting point is a univariate time series. From that, we extract an implicit x as lag 1. The x is a lagged value!
#       That's the reason why we must treat the time series as a univariate time series although we have a x.
#       But this x isn't an explicit x. Every non-lagged x is an explicit x and spawn a multivariate time series.

resample.features <- function(timeseries, timesteps, sep = "_") {
  df <- as.data.frame(timeseries)
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
    dflist[[i]] <- matrix(t(df[i:(i+timesteps-1),]), nrow = 1)
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

cross_validation_split <- function (dataset, folds = 3, foldname = "fold", random = FALSE) {
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
    # Further hidden layers
    i <- 1 # hidden layers
    d <- 1 # dropout layers to prevent overfitting
    D <- ifelse(!(is.null(dropout)),NROW(dropout),0)
    if (D > 0) {mlp_model %>% layer_dropout(rate = dropout[d]); d <- d + 1}
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

# Data format for LSTM
# X: Features must be in a 3D-array with following dimensionens
##   Samples  : Number of records
##   Timesteps: Number of different periods within a record (sample)
##   Features : Number of features (x) within a sequence of period
# Y: Outcomes must be in a 2D-array with the dimensions Samples and Units (number of output units)

as.LSTM.X <- function(x, features, timesteps){
  X.tensor <- data.matrix(x)
  X.tensor <- array(data = X.tensor, dim = c(NROW(X.tensor), timesteps, features))
  return(X.tensor)
}

as.LSTM.Y <- function(y, units){
  Y.tensor <- y
  Y.tensor <- array(data = Y.tensor, dim = c(NROW(Y.tensor), units))
  return(Y.tensor)
}

# Build LSTM architecture
# Univariate time series  : usually stateful = T and batchsize = 1
# Multivariate time series: usually stateful = F and batchsize = NULL; return_sequences = T
build_lstm <- function(features, timesteps = 1, batchsize = NULL, hidden, output = c(1,"linear"), 
                       stateful = FALSE, return_sequences = TRUE,
                       loss, optimizer, metrics) {
  lstm_model <- keras_model_sequential()
  h <- as.data.frame(hidden)
  N <- NROW(h)
  rs <- return_sequences
  # First hidden layer
  if (is.null(batchsize)) {
    lstm_model %>% layer_lstm(units = h[1,1], input_shape = c(timesteps, features), activation = h[1,2], stateful = stateful, return_sequences = rs)
  } else {
    lstm_model %>% layer_lstm(units = h[1,1], batch_input_shape = c(batchsize, timesteps, features), activation = h[1,2], stateful = stateful, return_sequences = rs)
  }
  # Further hidden layers
  i <- 1
  while (i < N) {
    if ((i == (N-1)) && (rs == T)) {rs <- !rs}
    lstm_model %>% layer_lstm(units = h[i+1,1], activation = h[i+1,2], stateful = stateful, return_sequences = rs)
    i <- i + 1
  }
  # Output layer
  lstm_model %>% layer_dense(units = output[1], activation = output[2])

  lstm_model %>% compile(loss = loss, optimizer = optimizer, metrics = metrics)
  return(lstm_model)
}

# Quality measures

## Mean Absolute Error
mae <- function (actual, predicted) {
  error <- actual - predicted
  return(mean(abs(error)))
}

## Mean Absolute Percentage Error
mape <- function(actual, predicted){
  error <- actual - predicted
  return(mean(abs(error/actual))*100)
}

## Root Mean Square Error
rmse <- function (actual, predicted) {
  error <- actual - predicted
  return(sqrt(mean(error^2)))
}

## Variance Coefficient
vc <- function (actual, predicted) {
  error <- actual - predicted
  return(sqrt(mean(error^2))/mean(actual))
}