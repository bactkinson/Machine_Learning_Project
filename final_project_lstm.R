rm(list=ls())
#LIBRARIES
library(quantmod)
library(tidyquant)
library(tibble)
library(fs)
library(jsonlite)
library(httr)
library(XML)
library(rvest)
library(ggplot2)
library(dplyr)
library(keras)
library(rsample)
library(recipes)
library(Metrics)
library(glue)
library(tidyr)
library(tensorflow)
# FUNCTION DEFINITIONS
calc_rmse <- function(finalized_df){
  duplicate.locs <- duplicated(finalized_df$date) | duplicated(finalized_df$date,fromLast=TRUE)
  
  duplicates <- finalized_df[duplicate.locs,]
  
  actuals <- duplicates %>%
    group_split(key) %>%
    .[[1]] %>%
    .['close'] %>%
    unlist(use.names = FALSE)
  
  predictions <- duplicates %>%
    group_split(key) %>%
    .[[2]] %>%
    .['close'] %>%
    unlist(use.names = FALSE)
  
  rmse_val <- Metrics::rmse(actuals,predictions)
  
  return(round(rmse_val,2))
}

create_lstm_tbl <- function(input.data,lag.vec,stock.name,key.type,...){
  lag.vec <- sort(lag.vec,decreasing = FALSE)
  
  for (i in 1:length(lag.vec)){
    n <- lag.vec[i]
    input.data <- input.data %>%
      mutate("{stock.name}_Lag_{n}" := lag(close,lag.vec[i]))
  }
  
  lstm_tbl <- input.data %>%
    filter(!(is.na("Lag_{n}"))) %>%
    filter(key==key.type) %>%
    {if(key.type=="Training") tail(.,...[[1]]) else .}
  return(lstm_tbl)
}

weighted_mse <- function( y_true, y_pred ) {
  K <- backend()
  K$mean(K$square(K.weights*(K$tanh(y_true)-K$tanh(y_pred))))
}

fit_LSTM_to_singlestock <- function(split,keras_model,lag_setting,batch_size,train_length,tsteps_input,
                              epochs_input){
  
  lag.setting <- lag_setting # nrow(first.split.testing)
  batch.size <- batch_size 
  train.length <- train_length
  tsteps <- tsteps_input
  epochs <- epochs_input
  
  if(tsteps != length(lag.setting)){
    warning("Number of time steps not equal to number of input lags")
  }
  
  split.training <- split %>%
    training() %>%
    add_column(key = "Training")
  
  split.testing <- split %>%
    testing() %>%
    add_column(key = "Testing")
  
  overall.split <- bind_rows(split.training,split.testing)
  
  p1 <- ggplot(data=overall.split,aes(date,close,color=key)) +
    geom_point() + 
    labs(title = "Testing and Training Data First Split",
         x = "Time",
         y = "Closing Price (in US $)") + 
    theme_classic() + 
    theme(legend.position = "bottom",legend.title = element_blank())
  
  rec_obj <- recipe(close ~ .,overall.split) %>%
    step_center(close) %>%
    step_scale(close) %>%
    prep()
  
  centered.data <- bake(rec_obj,overall.split)
  
  original.center <- rec_obj$steps[[1]]$means["close"]
  original.deviation <- rec_obj$steps[[2]]$sds["close"]
  
  lag_train_tbl <- create_lstm_tbl(centered.data,lag.setting,"AAPL","Training",train.length)
  
  x_train_data <- lag_train_tbl %>%
    select(contains("Lag")) %>%
    as.matrix(nrow=nrow(x_train_data),ncol=ncol(x_train_data))
  x_train_arr <- array(data=x_train_data,dim=c(dim(x_train_data),1))
  
  y_train_vec <- lag_train_tbl$close
  y_train_arr <- array(data = y_train_vec, dim=c(length(y_train_vec),1))
  
  lag_test_tbl <- create_lstm_tbl(centered.data,lag.setting,"AAPL","Testing")
  
  x_test_data <- lag_test_tbl %>%
    select(contains("Lag")) %>%
    as.matrix(nrow=nrow(x_test_data),ncol=ncol(x_test_data))
  x_test_arr <- array(data=x_test_data,c(dim(x_test_data),1))
  
  y_test_vec <- lag_test_tbl$close
  y_test_arr <- array(data=y_test_vec,dim=c(length(y_test_vec),1))
  
  for (i in 1:epochs) {
    keras_model %>% fit(x=x_train_arr,
                        y=y_train_arr,
                        batch_size=batch.size,
                        epochs=1,
                        verbose=1,
                        shuffle=FALSE)
    
    keras_model %>% reset_states()
    cat("Epoch: ",i)
  }
  
  ## Execute predictions on the test set
  closing_pred <- keras_model %>%
    predict(x_test_arr, batch_size = batch.size) %>%
    .[,1] %>%
    as_tibble() %>%
    mutate(s1=value*original.deviation,.keep="unused") %>%
    mutate(close = s1+original.center,.keep = "unused") %>%
    cbind("date"=lag_test_tbl$date) %>%
    add_column(key="Prediction")
  
  ## Combine predictions with original split values to create overall set.
  final.data.set <- overall.split %>%
    select(date,close,key) %>%
    mutate(key = "Actual",.keep = "unused") %>%
    bind_rows(closing_pred)
  
  rmse_evaluation <- calc_rmse(final.data.set)
  
  ## Calcualte the RMSE between predictions, 
  ## Plot the results
  p2 <- ggplot(data=final.data.set,aes(date,close,color=key)) +
    geom_point(alpha = 0.5) +
    scale_color_manual(values = c("black","red")) + 
    labs(title = paste0("Closing Price Time Series with RMSE: ",rmse_evaluation),
         x = "Time",
         y = "Closing Price") + 
    theme_classic() + 
    theme(legend.title = element_blank())  
  
  p2
  
  return(list(rmse_evaluation,p1,p2))
}

fit_LSTM_to_multistocks <- function(split,keras_model,lag_setting,batch_size,train_length,
                                    num_features,tsteps_input,epochs_input){
  
  lag.setting <- lag_setting # nrow(first.split.testing)
  batch.size <- batch_size 
  train.length <- train_length
  num.features <- num_features
  tsteps <- tsteps_input
  epochs <- epochs_input
  
  if(tsteps != length(lag.setting)){
    warning("Number of time steps not equal to number of input lags")
  }
  
  split.training <- split %>%
    training() %>%
    add_column(key = "Training")
  
  split.testing <- split %>%
    testing() %>%
    add_column(key = "Testing")
  
  overall.split <- bind_rows(split.training,split.testing)
  
  rec_obj <- recipe(key ~ .,data=overall.split) %>%
    update_role(date,new_role = "id") %>%
    step_center(all_predictors()) %>%
    step_scale(all_predictors()) %>%
    prep()
  
  centered.data <- bake(rec_obj,overall.split)
  
  original.center <- rec_obj$steps[[1]]$means[1]
  original.deviation <- rec_obj$steps[[2]]$sds[1]
  
  col_names <- names(centered.data)[2:(num.features+1)]
  
  lag_train_tbl <- tibble(rep(1,train.length))
  
  for(i in 1:num.features){
    current.stock <- centered.data %>%
      select(1,i+1,ncol(centered.data)) %>%
      rename("close"=col_names[i]) %>%
      create_lstm_tbl(.,lag.setting,col_names[i],"Training",train.length)
      
    lag_train_tbl <- cbind(lag_train_tbl,current.stock)
  }
  
  x_train_data <- lag_train_tbl %>%
    select(contains("Lag")) %>%
    as.matrix(nrow=nrow(x_train_data),ncol=ncol(x_train_data))
  x_train_arr <- array(data=x_train_data,dim=c(train.length,tsteps,num.features))
  
  y_train_vec <- lag_train_tbl[,3]
  y_train_arr <- array(data = y_train_vec, dim=c(length(y_train_vec),1))
  
  lag_test_tbl <- tibble(rep(1,nrow(split.testing)))
  
  for(i in 1:num.features){
    current.stock <- centered.data %>%
      select(1,i+1,ncol(centered.data)) %>%
      rename("close"=col_names[i]) %>%
      create_lstm_tbl(.,lag.setting,col_names[i],"Testing")
    
    lag_test_tbl <- cbind(lag_test_tbl,current.stock)
  }
    
  x_test_data <- lag_test_tbl %>%
    select(contains("Lag")) %>%
    as.matrix(nrow=nrow(x_test_data),ncol=ncol(x_test_data))
  x_test_arr <- array(data=x_test_data,c(nrow(split.testing),tsteps,num.features))
  
  y_test_vec <- lag_test_tbl[,3]
  y_test_arr <- array(data=y_test_vec,dim=c(length(y_test_vec),1))
  
  for (i in 1:epochs) {
    keras_model %>% fit(x=x_train_arr,
                        y=y_train_arr,
                        batch_size=batch.size,
                        epochs=1,
                        verbose=1,
                        shuffle=FALSE)
    
    keras_model %>% reset_states()
    cat("Epoch: ",i)
  }
  
  ## Execute predictions on the test set
  closing_pred <- keras_model %>%
    predict(x_test_arr, batch_size = batch.size) %>%
    .[,1] %>%
    as_tibble() %>%
    mutate(s1=value*original.deviation,.keep="unused") %>%
    mutate(close = s1+original.center,.keep = "unused") %>%
    cbind("date"=lag_test_tbl$date) %>%
    add_column(key="Prediction")
  
  ## Combine predictions with original split values to create overall set.
  final.data.set <- overall.split %>%
    select(date,AAPL,key) %>%
    rename(close=AAPL) %>%
    mutate(key = "Actual",.keep = "unused") %>%
    bind_rows(closing_pred)
  
  rmse_evaluation <- calc_rmse(final.data.set)
  
  ## Calcualte the RMSE between predictions, actual values. 
  ## Plot the results
  p2 <- ggplot(data=final.data.set,aes(date,close,color=key)) +
    geom_point(alpha = 0.5) +
    scale_color_manual(values = c("black","red")) + 
    labs(title = paste0("Closing Price Time Series with RMSE: ",rmse_evaluation),
         x = "Time",
         y = "Closing Price") + 
    theme_classic() + 
    theme(legend.title = element_blank())  

  return(list(rmse_evaluation,p2))
}
ts.predict <- function(data){
  # Compute the returns for the stock
  stock = data
  stock = stock[!is.na(stock)]
  breakpoint = floor(nrow(stock)*(2.9/3))
  
  # Initialzing a dataframe for the forecasted return series
  model.parameter <- arimaorder(auto.arima(stock, lambda = "auto"))
  p <- as.numeric(model.parameter[1])
  d <- as.numeric(model.parameter[2])
  q <- as.numeric(model.parameter[3])
  forecasted_series <- c()
  for (b in breakpoint:(nrow(stock)-1)) {
    stock_train = stock[1:b,]
    
    # Summary of the ARIMA model using the determined (p,d,q) parameters
    fit = arima(stock_train, order = c(p, d, q),include.mean=TRUE, optim.control = list(maxit = 1000))
    arima.forecast = forecast(fit, h = 1,level=99)
    # Creating a series of forecasted returns for the forecasted period
    forecasted_series = c(forecasted_series,arima.forecast$mean[1])
  }
  y <- stock[(floor(nrow(stock)*(2.9/3))+1):nrow(stock),]
  g <- cbind(y,forecasted_series)
  p1 <- plot(g, main = paste(names(data)))
  for (i in 1:(nrow(g)-1)){
    g$difference.actual[i] <- as.numeric(g[,1][i+1])-as.numeric(g[,1][i])
    g$difference.predict[i] <- as.numeric(g[,2][i+1])-as.numeric(g[,2][i])
  }
  tf <- c()
  for (i in 1:nrow(g)){
    if (sign(g$difference.actual[i]) == sign(g$difference.predict[i])){
      tf[i] = 1
    } else{
      tf[i] = 0
    }
  }
  Accuracy.percentage <- sum(tf)/length(tf)*100
  rmse.value <- rmse(g$difference.actual, g$difference.predict)
  return(list(p1,Accuracy.percentage,rmse.value))
}
#Final project
##Extracting S&P500 data from the packages
stock_list_tbl <- tq_index("SP500") %>%
  select(symbol, company, weight) %>% 
  arrange(desc(weight))

symbol <- stock_list_tbl$symbol[c(1:5)]

stock_data <- tq_get(symbol, get = "stock.prices",
                     from = "2014-01-01",to= "2021-04-01")

stock_data <- stock_data %>% 
  select(symbol, date, close)

## Isolate the apple stock closing price
aapl.data <- stock_data %>%
  group_split(symbol) %>%
  .[[1]]

spread_data <- stock_data %>%
  spread(.,symbol,close)

acf(aapl.data$close,lag.max=1500,main="Apple Close Price ACF")
abline(v=90,col="blue")

## Fitting model to entirety of AAPL time series data

training.period <- round(0.70*nrow(aapl.data),0)
testing.period <- nrow(aapl.data)-training.period

aapl.partitioning <- rolling_origin(
  aapl.data,
  initial = training.period,
  assess = testing.period,
)

## LSTM parameters
lag.setting <- c(1,2,3,90) # nrow(first.split.testing)
batch.size <- testing.period # both testing.period/batch.size and train.length/batch.size whole nums
train.length <- testing.period*2
tsteps <- 4
epochs <- 200

new.model <- keras_model_sequential()

new.model %>%
  layer_lstm(units = 50,
             input_shape = c(tsteps,1),
             batch_size = batch.size,
             return_sequences = TRUE,
             stateful = TRUE) %>%
  layer_dropout(0.3) %>%
  layer_lstm(units = 50,
             return_sequences = FALSE,
             stateful = TRUE) %>%
  layer_dense(units = 1)

new.model %>%
  compile(loss = 'mse',optimizer = 'adam')

aapl.output <- fit_LSTM_to_singlestock(aapl.partitioning$splits[[1]],
                            new.model,
                            lag.setting,
                            batch.size,
                            train.length,
                            tsteps,
                            epochs)

aapl.output[[3]]

## Incorporating multiple features
all.splits <- rolling_origin(
  spread_data,
  initial = training.period,
  assess = testing.period,
)

num.features <- ncol(spread_data)-1
 
multi.model <- keras_model_sequential()

multi.model %>%
  layer_lstm(units = 50,
             input_shape = c(tsteps,num.features),
             batch_size = batch.size,
             return_sequences = TRUE,
             stateful = TRUE) %>%
  layer_dropout(0.3) %>%
  layer_lstm(units = 50,
             return_sequences = FALSE,
             stateful = TRUE) %>%
  layer_dense(units = 1)

multi.model %>%
  compile(loss = 'mse',optimizer = 'adam')

multi.model.output <- fit_LSTM_to_multistocks(all.splits$splits[[1]],
                            multi.model,
                            lag.setting,
                            batch.size,
                            train.length,
                            num.features,
                            tsteps,
                            epochs)

multi.model.output[[2]] + 
  labs(subtitle = "With 4 Tech Stocks as Additional Predictors")
# 
multi.model.output[[2]] +
  geom_line() +
  coord_cartesian(xlim=c(date("2020-12-31"),date("2021-04-01")),ylim=c(110,145))

weighted.model <- keras_model_sequential()

K.weights <- tf$Variable(c(1:547/547),tf$float32)

weighted.model %>%
  layer_lstm(units = 50,
             input_shape = c(tsteps,num.features),
             batch_size = batch.size,
             activation = "tanh",
             return_sequences = TRUE,
             stateful = TRUE) %>%
  layer_dropout(0.5) %>%
  layer_lstm(units = 50,
             activation = "tanh",
             return_sequences = FALSE,
             stateful = TRUE) %>%
  layer_dense(units = 1)

weighted.model %>%
  compile(loss = weighted_mse,optimizer = 'adam')

weighted.output <- fit_LSTM_to_multistocks(all.splits$splits[[1]],
                                  weighted.model,
                                  lag.setting,
                                  batch.size,
                                  train.length,
                                  num.features,
                                  tsteps,
                                  epochs)

weighted.output[[2]] +
  labs(subtitle = "With weighted loss function and dropout rate increased")

### calculate Accuracy Percentage 
# a <- output[[4]]
# b <- a[a$key == "Actual",]
# c <- a[a$key == "Prediction",]
# colnames(c)[2] <- "Prediction.close"
# d <- merge(x= b,y = c,by = "date", all.x = TRUE)
# e <- d[!is.na(d$Prediction.close),]
# g <- cbind(e$close,e$Prediction.close)
# g <- as.data.frame(g)
# for (i in 1:(nrow(g)-1)){
#     g$difference.actual[i] <- as.numeric(g[,1][i+1])-as.numeric(g[,1][i])
#     g$difference.predict[i] <- as.numeric(g[,2][i+1])-as.numeric(g[,2][i])
# }
# tf <- c()
# for (i in 1:nrow(g)){
#     if (sign(g$difference.actual[i]) == sign(g$difference.predict[i])){
#       tf[i] = 1
#     } else{
#       tf[i] = 0
#     }
#   }
# Accuracy.percentage <- sum(tf)/length(tf)*100


###Time series prediction
getSymbols('AAPL', from = "2014-01-01",to= "2021-03-31")
getSymbols('MSFT', from = "2014-01-01",to= "2021-03-31")
getSymbols('AMZN', from = "2014-01-01",to= "2021-03-31")
getSymbols('FB', from = "2014-01-01",to= "2021-03-31")
getSymbols('GOOG', from = "2014-01-01",to= "2021-03-31")
aapl <- AAPL[,4]
msft <- MSFT[,4]
amzn <- AMZN[,4]
fb <- FB[,4]
goog <- GOOG[,4]

a <- ts.predict(aapl)
b <- ts.predict(msft)
c <- ts.predict(amzn)
d <- ts.predict(fb)
e <- ts.predict(goog)

par(mfrow=c(2,1))
a[[1]]
b[[1]]
