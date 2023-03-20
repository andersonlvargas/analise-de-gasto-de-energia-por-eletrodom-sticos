####rascunhos####
####selecionando variáveis pela colinearidade

resultado_colinearidade

treino_t<- treino_completo[,c("T_out", "T1", 'T6')]

treino_rh <- treino_completo[, c("RH_1", "RH_2", "RH_5", "RH_6", "RH_8", "RH_9")]

treino_selecionado <- cbind(treino_t, treino_rh)

treino_selecionado <- cbind(treino_completo[1], treino_selecionado)

teste_t<- teste_completo[,c("T_out", "T1", 'T6')]

teste_rh <- teste_completo[, c("RH_1", "RH_2", "RH_5", "RH_6", "RH_8", "RH_9")]

teste_selecionado <- cbind(teste_t, teste_rh)

teste_selecionado <- cbind(teste_completo[1], teste_selecionado)


#### fazendo o (vif)

model_vif <- glm(Appliances ~ ., data = treino_selecionado)

vif(model_vif)

####fazendo a PCA para diminuir a dminesionalidade

pca_treino <- prcomp(treino[-c(1,2)], center = TRUE, scale. = TRUE)

treino_pca<- data.frame(Eletro = treino[,1], pca_treino$x[,1:2])

pca_teste <- prcomp(teste[-c(1,2)], center = TRUE, scale. = TRUE)

teste_pca<- data.frame(Eletro = teste[,1], pca_teste$x[,1:2])


####carregando pacotes####
library(dplyr)
library(caret)
library(car)
library(ggplot2)
library(readr)
library(mltools)
library(data.table)
library(corrplot)
library(Metrics)
library(xgboost)
library(randomForest)
library(vegan)
library(e1071)

set.seed(1)

####carregando dados####

df <- read.csv("C:/Users/ander/Documents/r curso data science/projetos/Projetos-Modelagem Preditiva em IoT/projeto8-data_files/projeto8-training.csv")

colnames(df)

summary(df)
head(df)
dim(df)
str(df)
View(df)
names(df)

eliminar <- c("date", "lights")

eliminar

df[, which(names(df) %in% c(eliminar))] <- NULL


####vendo se tem na####

nas<- apply(is.na(df), 2, sum)

nas

####transformando variaveis em fator e numericos####

df[,c('WeekStatus','Day_of_week')] <- lapply(df[,c('WeekStatus','Day_of_week')], as.factor)


df[,c('Appliances')] <- lapply(df[,c('Appliances')], as.numeric)

####normalizando os dados de entrada#####

df$Appliances <- scale(df$Appliances)


####fazendo dados teste e treino####

numero_linhas<- length(rownames(df))

numero_treino <- round( 0.8 * numero_linhas)

linhas <- sample(1:numero_linhas, numero_treino, replace = FALSE)

treino_completo <- df[linhas,]

teste_completo <- df[-linhas,]


####fazendo o label encoding#### 

treino_completo <- treino_completo %>% mutate_if(is.character, as.factor) %>% mutate_if(is.factor, as.numeric)

teste_completo <- teste_completo %>% mutate_if(is.character, as.factor) %>% mutate_if(is.factor, as.numeric)

####vendo a correlação entre as variáveis e colinearidade perfeita####

corre <-cor(treino_completo[-c(1,2)], method = "pearson")
x11()
corrplot(corre)

if (any(corre == 1 | corre == -1)) {
  print("Há colinearidade perfeita entre as variáveis")
} else {
  print("Não há colinearidade perfeita entre as variáveis")
}

colnames(corre)[1]

colinearidade <- function(x, v) {
  valores <- vector()
  colunas <- vector()
  linhas <- vector()
  for (l in 1:length(rownames(x))) {
    for (c in 1:length(colnames(x))) {
      if (x[l,c] > v & colnames(x)[c] != rownames(x) [l] ) {
        valores <- c(valores,(x[l,c]))
        colunas <- c(colunas, colnames(x)[c] )
        linhas <- c(linhas, rownames(x) [l])
      }
    }
  }
  resultado <- data.frame(colunas = colunas, linhas = linhas, valores = valores)
}

resultado_colinearidade <- colinearidade(corre, 0.8)

resultado_colinearidade <- subset(resultado_colinearidade
                                  , colunas != linhas)

####selecionando variáveis para eletrodomésticos treino por random forest####

variaveis = randomForest(Appliances ~ .,data = treino_completo[-2], mtry = 2,ntree=500, importance = TRUE)

x11()
varImpPlot(variaveis, sort = T)

importancia_variaveis <- importance(variaveis)

importancia_variaveis_ordenada <- importancia_variaveis[order(importancia_variaveis[, "IncNodePurity"], decreasing = TRUE), ]

variaveis_eletro <-importancia_variaveis_ordenada[1:which(row.names(importancia_variaveis_ordenada) == "Windspeed"),]

treino <- treino_completo[,c('Appliances',row.names(variaveis_eletro))]

teste <- teste_completo[,c("Appliances", row.names(variaveis_eletro))]

##match("day_of_week", names(importancia_variaveis_ordenada))# é para coluna

####fazendo o VIF nos dados de random forest####
model_vif <- glm(Appliances ~ ., data = treino)

VIF_random <- vif(model_vif)

variaveis_vif <- c("Appliances","NSM", "Press_mm_hg", "RH_8", "RH_9", "T8", "RH_5",
                   "T4", "Windspeed")

treino_vif <- treino[, c(variaveis_vif)]

teste_vif <- teste[, c(variaveis_vif)]

####fazendo o modelo XGboost####

# Definindo os parâmetros iniciais do modelo
xgb_params <- list(
  objective = "reg:squarederror",
  eval_metric = "rmse"
)

# Realizando o tunning dos parâmetros do modelo
xgb_grid <- expand.grid(
  nrounds = c(100, 150, 200),
  max_depth = c(6, 8, 10),
  colsample_bytree = c(0.6, 0.8, 1),
  eta = c(0.01, 0.1, 0.3),
  gamma = 0,
  min_child_weight = 1,
  subsample = 1
)


xgb_tune <- train(
  Appliances ~ .,
  data = treino_selecionado[-2],
  method = "xgbTree",
  trControl = trainControl(
    method = "cv",
    number = 5,
    verboseIter = TRUE
  ),
  tuneGrid = xgb_grid
)

# Exibindo os melhores parâmetros encontrados
xgb_tune$bestTune

# Treinando o modelo com os melhores parâmetros encontrados
xgb_model <- xgboost(
  data = as.matrix(treino_vif),
  label = treino[, 1],
  nrounds = xgb_tune$bestTune$nrounds,
  max_depth = xgb_tune$bestTune$max_depth,
  colsample_bytree = xgb_tune$bestTune$colsample_bytree,
  eta = xgb_tune$bestTune$eta,
  params = xgb_params
)

# Realizando previsões no conjunto de teste
preds <- predict(xgb_model, as.matrix(teste_vif))

# Avaliando o modelo
mae(teste_vif$Appliances, preds)

mse(teste_vif$Appliances, preds)

mad(teste_vif$Appliances, preds)

R2(teste_vif$Appliances, preds)

