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
library(googleLanguageR)
library(sentimentr)
library(keras)
library(rsample)
library(recipes)
library(Metrics)
library(glue)
library(tidyr)
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

get.ny <- function(a){
  NYTIMES_KEY = "LXrkrR8vlr0RnqKwoA7we3dyCngAdEPk"
  
  x <- fromJSON(paste0("http://api.nytimes.com/svc/search/v2/articlesearch.json?q=",a,"&api-key=LXrkrR8vlr0RnqKwoA7we3dyCngAdEPk", sep=""), flatten = TRUE) %>% data.frame()
  
  list(name = c("persons", "organizations", "subject"), value = c("Stock"), rank = 1:3, major = c("N", "N", "N"))
  # Need to use + to string together separate words
  begin_date <- "20190101"
  end_date <- "20210401"
  baseurl <- paste0("http://api.nytimes.com/svc/search/v2/articlesearch.json?q=",a,
                    "&begin_date=",begin_date,"&end_date=",end_date,
                    "&facet_filter=true&api-key=",NYTIMES_KEY, sep="")
  
  initialQuery <- fromJSON(baseurl)
  maxPages <- round((initialQuery$response$meta$hits[1] / 10)-1) 
  pages <- list()
  for(i in 0:3){
    nytSearch <- fromJSON(paste0(baseurl, "&page=", i), flatten = TRUE) %>% data.frame() 
    message("Retrieving page ", i)
    pages[[i+1]] <- nytSearch 
    Sys.sleep(10) 
  }
  allNYTSearch <- rbind_pages(pages)
  return(allNYTSearch)
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
         subtitle = "With 4 Tech Stocks as Additional Predictors",
         x = "Time",
         y = "Closing Price") + 
    theme_classic() + 
    theme(legend.title = element_blank())  
  
  p2
  
  return(list(rmse_evaluation,p2))
}
#Final project
##Extracting S&P500 data from the packages
stock_list_tbl <- tq_index("SP500") %>%
  select(symbol, company, weight) %>% 
  arrange(desc(weight))

symbol <- stock_list_tbl$symbol[c(1:5)]

stock_data <- tq_get(symbol, get = "stock.prices",
                     from = "2014-01-01",to= "2021-03-31")

stock_data <- stock_data %>% 
  select(symbol, date, close)

# company.names <- stock_list_tbl$company[c(1:5)]
# ny.data <- data.frame()
# ny.data <- rbind(ny.data,get.ny(company.names[1]))
# ny.data <- rbind(ny.data,get.ny(company.names[2]))
# ny.data <- rbind(ny.data,get.ny(company.names[3]))
# ny.data <- rbind(ny.data,get.ny(company.names[4]))
# ny.data <- rbind(ny.data,get.ny(company.names[5]))




# sentiment.score <- c()
# for (i in 1:nrow(data)){
#   sa <- sentiment_by(data$response.docs.abstract[i])
#   sentiment.score <- c(sentiment.score, sa$ave_sentiment)
# }
# sentiment.score


## Isolate the apple stock closing price
aapl.data <- stock_data %>%
  group_split(symbol) %>%
  .[[1]]

spread_data <- stock_data %>%
  spread(.,symbol,close)
## Create 4 splits to evaluate LSTM RNN separately to get sense of how well model performs
## across different samples (avoid inferring too much from single result)

training.period <- 720
testing.period <- 180
skip.span <- 180

backtest.splits <- rolling_origin(
  aapl.data,
  initial = training.period,
  assess = testing.period,
  cumulative = FALSE,
  skip = skip.span
)

## LSTM parameters
lag.setting <- c(1,2,3,90) # nrow(first.split.testing)
batch.size <- 30 # both testing.period/batch.size and train.length/batch.size whole nums
train.length <- 630
tsteps <- 4
epochs <- 200
# 
model <- keras_model_sequential()

model %>%
  layer_lstm(units = 50,
             input_shape = c(tsteps,1),
             batch_size = batch.size,
             return_sequences = TRUE,
             stateful = TRUE) %>%
  layer_lstm(units = 50,
             return_sequences = FALSE,
             stateful = TRUE) %>%
  layer_dense(units = 1)

model %>%
  compile(loss = 'mae',optimizer = 'adam')
 


## Try fitting model to entire aapl time series data

# training.period <- round(0.70*nrow(aapl.data),0)
# testing.period <- nrow(aapl.data)-training.period
# 
# 
# entire.partitioning <- rolling_origin(
#   aapl.data,
#   initial = training.period,
#   assess = testing.period,
# )
# 
# ## LSTM parameters
# lag.setting <- c(1,2,3,90) # nrow(first.split.testing)
# batch.size <- testing.period # both testing.period/batch.size and train.length/batch.size whole nums
# train.length <- testing.period*2
# tsteps <- 4
# epochs <- 200
# 
# new.model <- keras_model_sequential()
# 
# new.model %>%
#   layer_lstm(units = 50,
#              input_shape = c(tsteps,1),
#              batch_size = batch.size,
#              return_sequences = TRUE,
#              stateful = TRUE) %>%
#   layer_lstm(units = 50,
#              return_sequences = FALSE,
#              stateful = TRUE) %>%
#   layer_dense(units = 1)
# 
# new.model %>%
#   compile(loss = 'mse',optimizer = 'adam')
# 
# entire.output <- fit_LSTM_to_singlestock(entire.partitioning$splits[[1]],
#                             new.model,
#                             lag.setting,
#                             batch.size,
#                             train.length,
#                             tsteps,
#                             epochs)
# 
# entire.output[[3]]

## Incorporating multiple features
new.splits <- rolling_origin(
  spread_data,
  initial = training.period,
  assess = testing.period,
  cumulative = FALSE,
  skip = skip.span
)

lag.setting <- c(1,2,3,90) # nrow(first.split.testing)
batch.size <- 30 # both testing.period/batch.size and train.length/batch.size whole nums
train.length <- 630
tsteps <- 4
epochs <- 200
num.features <- ncol(spread_data)-1
# 
multi.model <- keras_model_sequential()

multi.model %>%
  layer_lstm(units = 50,
             input_shape = c(tsteps,num.features),
             batch_size = batch.size,
             return_sequences = TRUE,
             stateful = TRUE) %>%
  layer_lstm(units = 50,
             return_sequences = FALSE,
             stateful = TRUE) %>%
  layer_dense(units = 1)

multi.model %>%
  compile(loss = 'mae',optimizer = 'adam')

multi.model.output <- fit_LSTM_to_multistocks(new.splits$splits[[6]],
                            multi.model,
                            lag.setting,
                            batch.size,
                            train.length,
                            num.features,
                            tsteps,
                            epochs)

multi.model.output[[2]]

split.sequestered <- backtest.splits$splits[[6]]
output <- fit_LSTM_to_singlestock(split.sequestered,
                                  model,
                                  lag.setting,
                                  batch.size,
                                  train.length,
                                  tsteps,
                                  epochs)
# To visualize plot, call the 3rd element of the list
output[[3]]
