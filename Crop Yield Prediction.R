library(ggplot2)
library(ggthemes)
library(Amelia)
library(plotly)
library(corrplot)
library(GGally)
library(caTools)
library(glmnet)
library(e1071)
library(FNN)
library(dplyr)
library(rpart)
library(rpart.plot)
library(randomForest)
library(xgboost)
library(nnet)
library(doParallel)
library(brms)
library(NeuralNetTools)
library(lightgbm)
library(quantreg)
library(reticulate)
library(mgcv)
library(GauPro)
library(caret)
options(brms.backend = "cmdstanr")

cl <- makeCluster(detectCores() - 1)  # Leave one core free
registerDoParallel(cl)

set.seed(101)
df <- read.csv('crop_yield.csv')
df <- df[sample(nrow(df),0.04 * nrow(df)),]
dim(df)
str(df)
summary(df)

any(is.na(df))
any(duplicated(df))

missmap(df,main='Missingness Map',col=c('yellow','black'),y.at=0,y.labels=NULL)

# Convert categorical variables into factors
df$Soil_Type <- factor(df$Soil_Type)
df$Weather_Condition <- factor(df$Weather_Condition)
df$Crop <- factor(df$Crop)
df$Region <- factor(df$Region)
df$Fertilizer_Used <- factor(df$Fertilizer_Used)
df$Irrigation_Used <- factor(df$Irrigation_Used)

# Exploratory Data Analysis (EDA)

# Univariate Analysis
ggplotly(ggplot(data=df,mapping=aes(x=Rainfall_mm)) + geom_histogram(bins=30,alpha=0.7,color='black',fill='purple') + labs(x='Rainfall (mm)',y='Frequency',title='Distribution of Rainfall') + theme_bw() + theme(plot.title=element_text(hjust=0.5,face='bold')))
ggplotly(ggplot(data=df,mapping=aes(x=Temperature_Celsius)) + geom_histogram(bins=30,alpha=0.7,color='black',fill='sienna') + labs(x='Temperature (celsius)',y='Frequency',title='Distribution of Temperature') + theme_bw() + theme(plot.title=element_text(hjust=0.5,face='bold')))
ggplotly(ggplot(data=df,mapping=aes(x="",y=Days_to_Harvest)) + geom_boxplot(fill="skyblue", color="darkblue", outlier.color="red", outlier.shape=16, outlier.size=2) + stat_summary(fun=mean,geom="point",shape=20,size=3,color="black",fill="white") + labs(title="Distribution of Days to Harvest", y="Days to Harvest", x=NULL) + theme_bw() + theme(plot.title=element_text(hjust=0.5,size=14,face="bold"),axis.text.x=element_blank(),axis.ticks.x=element_blank()))
ggplotly(ggplot(data=df,mapping=aes(x=Yield_tons_per_hectare)) + geom_histogram(bins=30,alpha=0.7,color='black',fill='steelblue') + labs(x='Yield (tons/hectare)',y='Frequency',title='Distribution of Crop Yield') + theme_bw() + theme(plot.title=element_text(hjust=0.5,face='bold')))
prop.table(table(df$Region))
ggplotly(ggplot(data=df,mapping=aes(x=Region)) + geom_bar(stat='count',alpha=0.8,color='black',fill='pink') + labs(title='Countplot of Region') + theme_bw() + theme(plot.title=element_text(hjust=0.5,face='bold')))
prop.table(table(df$Weather_Condition))
ggplotly(ggplot(data=df,mapping=aes(x=Weather_Condition)) + geom_bar(stat='count',alpha=0.8,color='black',fill='violet') + labs(title='Countplot of Weather Condition') + theme_bw() + theme(plot.title=element_text(hjust=0.5,face='bold')))
prop.table(table(df$Soil_Type))
ggplotly(ggplot(data=df,mapping=aes(x=Soil_Type)) + geom_bar(stat='count',alpha=0.8,color='black',fill='lawngreen') + labs(title='Countplot of Soil Type') + theme_bw() + theme(plot.title=element_text(hjust=0.5,face='bold')))
prop.table(table(df$Crop))
ggplotly(ggplot(data=df,mapping=aes(x=Crop)) + geom_bar(stat='count',alpha=0.8,color='black',fill='cyan') + labs(title='Countplot of Crop') + theme_bw() + theme(plot.title=element_text(hjust=0.5,face='bold')))

# Bivariate Analysis
ggplotly(ggplot(data=df,mapping=aes(x=Region,y=Days_to_Harvest,colour=Weather_Condition)) + geom_bar(mapping=aes(fill=Weather_Condition),position='fill',alpha=0.7,stat='identity') + labs(title='Days to Harvest by Region & Weather Condition') + theme_bw() + theme(plot.title=element_text(hjust=0.5,face='bold')))
ggplotly(ggplot(data=df,mapping=aes(x=Days_to_Harvest,y=Yield_tons_per_hectare,color=Fertilizer_Used)) + geom_point(alpha=0.5,size=3) + scale_color_manual(values=c('blue','green')) + theme_classic()) 
ggplotly(ggplot(data=df,mapping=aes(x=Temperature_Celsius,y=Yield_tons_per_hectare,color=Fertilizer_Used)) + geom_point(alpha=0.6,size=3) + theme_bw())
ggplotly(ggplot(data=df,mapping=aes(x=Rainfall_mm,y=Yield_tons_per_hectare,color=Irrigation_Used)) + geom_point(alpha=0.6,size=3) + scale_color_manual(values=c('yellow','orangered')) + theme_bw())
# There is a strong positive correlation between rainfall and crop yield. 
ggplotly(ggplot(data=df,mapping=aes(x=Soil_Type,y=Days_to_Harvest,color=Fertilizer_Used)) + geom_col(alpha=0.7,position='fill') + labs(title='Days to Harvest by Soil Type & Fertilizer Used') + theme_economist_white() + theme(plot.title=element_text(hjust=0.5,face='bold')))

ggplotly(ggplot(data=df,mapping=aes(x=Soil_Type,y=Yield_tons_per_hectare,fill=Fertilizer_Used)) + geom_boxplot(mapping=aes(fill=Fertilizer_Used),alpha=0.7,position='dodge',color='orange') + labs(title='Crop Yield by Soil Type & Fertilizer') + theme_bw() + theme(plot.title=element_text(hjust=0.5,face='bold')))
ggplotly(ggplot(data=df,mapping=aes(x=Crop,y=Days_to_Harvest,color=Fertilizer_Used)) + geom_col(mapping=aes(color=Fertilizer_Used),alpha=0.7,position='fill') + labs(title='Days to Harvest by Crop & Fertilizer') + theme_bw() + theme(plot.title=element_text(hjust=0.5,face='bold')))                                                                                                                                                                                                                                        
ggplotly(ggplot(data=df,mapping=aes(x=Crop,y=Yield_tons_per_hectare,color=Fertilizer_Used)) + geom_violin(mapping=aes(color=Fertilizer_Used),alpha=0.65,position='dodge') + labs(title='Yield by Crop & Fertilizer Used') + theme_bw() + theme(plot.title=element_text(hjust=0.5,face='bp;d')))
ggplotly(ggplot(data=df,mapping=aes(x=Yield_tons_per_hectare,fill=Fertilizer_Used)) + geom_density(alpha=0.5) + facet_grid(. ~ Region) + labs(title='Crop Yield by Fertilizer and Region') + theme_bw() + theme(plot.title=element_text(hjust=0.5,face='bold'))) 
ggplotly(ggplot(data=df,mapping=aes(x=Yield_tons_per_hectare,fill=Fertilizer_Used)) + geom_density(alpha=0.5) + facet_grid(. ~ Soil_Type) + labs(title='Crop Yield by Fertilizer and Soil Type') + scale_fill_manual(values=c('darkgreen','tomato')) + theme_bw() + theme(plot.title=element_text(hjust=0.5,face='bold'))) 

# Multivariate Analysis
# Compute correlation matrix
corr.matrix <- cor(df[sapply(df,is.numeric)],use="complete.obs")

corrplot(corr.matrix,method='color',type='full',col=colorRampPalette(c('red','green','blue'))(200),addCoef.col='black')

# Heatmap
scaled_df <- scale(df[sapply(df,is.numeric)])

heatmap(as.matrix(scaled_df),Rowv=NA,Colv=NA,col=terrain.colors(256),scale="column")

# Pairplot
ggplotly(ggpairs(df[sapply(df,is.numeric)]) + theme_bw())

# Model Training & Evaluation

# Train test split
set.seed(101)
split <- sample.split(Y=df$Yield_tons_per_hectare,SplitRatio=0.7)

train <- subset(df,split==T)
test <- subset(df,split==F)

dim(train)
dim(test)

# Create a dataframe to store model results
model.performance <- data.frame(
  model = character(),
  r2.score = numeric(),
  stringsAsFactors = FALSE
)

# Linear Regression
lr.model <- lm(formula=Yield_tons_per_hectare ~ .,data=train)
print(summary(lr.model))

lr.predictions <- predict(lr.model,newdata=test)

mse <- mean((test$Yield_tons_per_hectare - lr.predictions) ** 2)

# MAE
mae <- mean(abs(test$Yield_tons_per_hectare - lr.predictions))

# MAPE
mape <- mean(abs((test$Yield_tons_per_hectare - lr.predictions) / test$Yield_tons_per_hectare)) * 100

# RMSLE
rmsle <- sqrt(mean((log1p(lr.predictions) - log1p(test$Yield_tons_per_hectare))^2))

# R-squared
r2 <- summary(lr.model)$r.squared
adj.r2 <- summary(lr.model)$adj.r.squared

cat("LR MSE:", mse, "\n")
cat("LR RMSE:", sqrt(mse), "\n")
cat("LR R-squared:", r2, "\n")
cat("LR Adjusted R-squared:", adj.r2, "\n")
cat("LR MAE:", mae, "\n")
cat("LR MAPE (%):", round(mape, 2), "\n")
cat("LR RMSLE:", rmsle, "\n")

model.performance <- rbind(model.performance,data.frame(model="Linear Regression",r2.score=r2))

# Gaussian GLM Model
gaussian.glm.model <- glm(formula=Yield_tons_per_hectare ~ .,data=train,family='gaussian')
print(summary(gaussian.glm.model))

gaussian.predictions <- predict(gaussian.glm.model,newdata=test)

# MSE
mse <- mean((test$Yield_tons_per_hectare - gaussian.predictions) ** 2)

# MAE
mae <- mean(abs(test$Yield_tons_per_hectare - gaussian.predictions))

# MAPE
mape <- mean(abs((test$Yield_tons_per_hectare - gaussian.predictions) / test$Yield_tons_per_hectare)) * 100

# RMSLE
rmsle <- sqrt(mean((log1p(gaussian.predictions) - log1p(test$Yield_tons_per_hectare))^2))

# R-squared
sst <- sum((test$Yield_tons_per_hectare - mean(test$Yield_tons_per_hectare)) ** 2) # Sum of Squares Total (SST)
sse <- sum((test$Yield_tons_per_hectare - gaussian.predictions) ** 2) # Sum of Squares Error (SSE)

r2 <- 1 - sse / sst

# Adjusted R-squared
n <- nrow(test)       # Number of observations
p <- length(coef(gaussian.glm.model)) - 1  # Number of predictors (excluding intercept)
adj.r2 <- 1 - (1 - r2) * ((n - 1) / (n - p - 1))

print(paste("Gaussian GLM MSE:", mse))
print(paste("Gaussian GLM RMSE:", sqrt(mse)))
print(paste("Gaussian GLM MAE:", mae))
print(paste("Gaussian GLM MAPE:", mape))
print(paste("Gaussian GLM RMSLE:", rmsle))
print(paste("Gaussian GLM R-squared:", round(r2, 4)))
print(paste("Gaussian GLM Adjusted R-squared:", round(adj.r2, 4)))

model.performance <- rbind(model.performance,data.frame(model="Gaussian GLM",r2.score=r2))

# Ridge regression
x_train <- model.matrix(Yield_tons_per_hectare ~ .,data=train)[,-1]
y_train <- as.numeric(train$Yield_tons_per_hectare)
x_test <- model.matrix(Yield_tons_per_hectare ~ .,data=test)[,-1]
y_test <- as.numeric(test$Yield_tons_per_hectare)

ridge.model <- glmnet(x_train,y_train,alpha=0)
plot(ridge.model)

ridge.predictions <- predict(ridge.model, newx = x_test, s = ridge.model$lambda.min)

# MSE
mse <- mean((test$Yield_tons_per_hectare - ridge.predictions) ** 2)

# MAE
mae <- mean(abs(test$Yield_tons_per_hectare - ridge.predictions))

# MAPE
mape <- mean(abs((test$Yield_tons_per_hectare - ridge.predictions) / test$Yield_tons_per_hectare)) * 100

# RMSLE
rmsle <- sqrt(mean((log1p(ridge.predictions) - log1p(test$Yield_tons_per_hectare))^2))

# R-squared
sst <- sum((test$Yield_tons_per_hectare - mean(test$Yield_tons_per_hectare)) ** 2) # Sum of Squares Total (SST)
sse <- sum((test$Yield_tons_per_hectare - ridge.predictions) ** 2) # Sum of Squares Error (SSE)

r2 <- 1 - sse / sst

# Adjusted R-squared
n <- nrow(test)       # Number of observations
p <- sum(coef(ridge.model, s = ridge.model$lambda.min) != 0) - 1  # Number of predictors (excluding intercept)
adj.r2 <- 1 - (1 - r2) * ((n - 1) / (n - p - 1))

print(paste("Ridge MSE:", mse))
print(paste("Ridge RMSE:", sqrt(mse)))
print(paste("Ridge MAE:", mae))
print(paste("Ridge MAPE:", mape))
print(paste("Ridge RMSLE:", rmsle))
print(paste("Ridge R-squared:", round(r2, 4)))
print(paste("Ridge Adjusted R-squared:", round(adj.r2, 4)))

model.performance <- rbind(model.performance,data.frame(model="Ridge",r2.score=r2))

# Lasso regression
lasso.model <- glmnet(x_train,y_train,alpha=1)
plot(lasso.model)

lasso.predictions <- predict(lasso.model,newx=x_test,s=lasso.model$lambda.min)

# MSE
mse <- mean((test$Yield_tons_per_hectare - lasso.predictions) ** 2)

# MAE
mae <- mean(abs(test$Yield_tons_per_hectare - lasso.predictions))

# MAPE
mape <- mean(abs((test$Yield_tons_per_hectare - lasso.predictions) / test$Yield_tons_per_hectare)) * 100

# RMSLE
rmsle <- sqrt(mean((log1p(lasso.predictions) - log1p(test$Yield_tons_per_hectare))^2))

# R-squared
sst <- sum((test$Yield_tons_per_hectare - mean(test$Yield_tons_per_hectare)) ** 2) # Sum of Squares Total (SST)
sse <- sum((test$Yield_tons_per_hectare - lasso.predictions) ** 2) # Sum of Squares Error (SSE)

r2 <- 1 - sse / sst

# Adjusted R-squared
n <- nrow(test)       # Number of observations
p <- sum(coef(lasso.model, s = lasso.model$lambda.min) != 0) - 1  # Number of predictors (excluding intercept)
adj.r2 <- 1 - (1 - r2) * ((n - 1) / (n - p - 1))

print(paste("Lasso MSE:", mse))
print(paste("Lasso RMSE:", sqrt(mse)))
print(paste("Lasso MAE:", mae))
print(paste("Lasso MAPE:", mape))
print(paste("Lasso RMSLE:", rmsle))
print(paste("Lasso R-squared:", round(r2, 4)))
print(paste("Lasso Adjusted R-squared:", round(adj.r2, 4)))

model.performance <- rbind(model.performance,data.frame(model="Lasso",r2.score=r2))

# Elastic Net regression
enet.model <- glmnet(x_train,y_train,alpha=0.5)
plot(enet.model)

enet.predictions <- predict(enet.model,newx=x_test,s=enet.model$lambda.min)

# MSE
mse <- mean((test$Yield_tons_per_hectare - enet.predictions) ** 2)

# MAE
mae <- mean(abs(test$Yield_tons_per_hectare - enet.predictions))

# MAPE
mape <- mean(abs((test$Yield_tons_per_hectare - enet.predictions) / test$Yield_tons_per_hectare)) * 100

# RMSLE
rmsle <- sqrt(mean((log1p(enet.predictions) - log1p(test$Yield_tons_per_hectare))^2))

# R-squared
sst <- sum((test$Yield_tons_per_hectare - mean(test$Yield_tons_per_hectare)) ** 2) # Sum of Squares Total (SST)
sse <- sum((test$Yield_tons_per_hectare - enet.predictions) ** 2) # Sum of Squares Error (SSE)

r2 <- 1 - sse / sst

# Adjusted R-squared
n <- nrow(test)       # Number of observations
p <- sum(coef(enet.model, s = enet.model$lambda.min) != 0) - 1  # Number of predictors (excluding intercept)
adj.r2 <- 1 - (1 - r2) * ((n - 1) / (n - p - 1))

print(paste("Elastic Net MSE:", mse))
print(paste("Elastic Net RMSE:", sqrt(mse)))
print(paste("Elastic Net MAE:", mae))
print(paste("Elastic Net MAPE:", mape))
print(paste("Elastic Net RMSLE:", rmsle))
print(paste("Elastic Net R-squared:", round(r2, 4)))
print(paste("Elastic Net Adjusted R-squared:", round(adj.r2, 4)))

model.performance <- rbind(model.performance,data.frame(model="Elastic Net",r2.score=r2))

# Support Vector Machines (SVM)
svm.model <- svm(formula=Yield_tons_per_hectare ~ .,data=train)
print(summary(svm.model))

svm.predictions <- predict(svm.model,newdata=test)

# MSE
mse <- mean((test$Yield_tons_per_hectare - svm.predictions) ** 2)

# MAE
mae <- mean(abs(test$Yield_tons_per_hectare - svm.predictions))

# MAPE
mape <- mean(abs((test$Yield_tons_per_hectare - svm.predictions) / test$Yield_tons_per_hectare)) * 100

# RMSLE
rmsle <- sqrt(mean((log1p(svm.predictions) - log1p(test$Yield_tons_per_hectare))^2))

# R-squared
sst <- sum((test$Yield_tons_per_hectare - mean(test$Yield_tons_per_hectare)) ** 2) # Sum of Squares Total (SST)
sse <- sum((test$Yield_tons_per_hectare - svm.predictions) ** 2) # Sum of Squares Error (SSE)

r2 <- 1 - sse / sst

# Adjusted R-squared
n <- nrow(test)       # Number of observations
p <- length(svm.model) - 1  # Number of predictors (excluding intercept)
adj.r2 <- 1 - (1 - r2) * ((n - 1) / (n - p - 1))

print(paste("SVM MSE:", mse))
print(paste("SVM RMSE:", sqrt(mse)))
print(paste("SVM MAE:", mae))
print(paste("SVM MAPE:", mape))
print(paste("SVM RMSLE:", rmsle))
print(paste("SVM R-squared:", round(r2, 4)))
print(paste("SVM Adjusted R-squared:", round(adj.r2, 4)))

model.performance <- rbind(model.performance,data.frame(model="Support Vector Machines",r2.score=r2))

# K Nearest Neighbors
x_train <- model.matrix(~ . -1, data=subset(train,select=-Yield_tons_per_hectare))
y_train <- train$Yield_tons_per_hectare
x_test <- model.matrix(~ . -1, data=subset(test,select=-Yield_tons_per_hectare))

mins <- apply(x_train,2,min)
maxs <- apply(x_train,2,max)

# Scale the data
x_train <- scale(x_train,center=mins,scale=(maxs-mins))
x_test <- scale(x_test,center=mins,scale=(maxs-mins))

knn.model <- knn.reg(x_train,x_test,y_train,k=2)
print(summary(knn.model))

knn.predictions <- knn.model$pred

# MSE
mse <- mean((test$Yield_tons_per_hectare - knn.predictions) ** 2)

# RMSE
rmse <- sqrt(mse)

# MAE
mae <- mean(abs(test$Yield_tons_per_hectare - knn.predictions))

# MAPE
mape <- mean(abs((test$Yield_tons_per_hectare - knn.predictions) / test$Yield_tons_per_hectare)) * 100

# RMSLE
rmsle <- sqrt(mean((log1p(knn.predictions) - log1p(test$Yield_tons_per_hectare))^2))

# R-squared
sst <- sum((test$Yield_tons_per_hectare - mean(test$Yield_tons_per_hectare)) ** 2) # Sum of Squares Total (SST)
sse <- sum((test$Yield_tons_per_hectare - knn.predictions) ** 2) # Sum of Squares Error (SSE)

r2 <- 1 - sse / sst

# Adjusted R-squared
n <- nrow(test)       # Number of observations
p <- length(knn.model) - 1  # Number of predictors (excluding intercept)
adj.r2 <- 1 - (1 - r2) * ((n - 1) / (n - p - 1))

print(paste("KNN MSE:", mse))
print(paste("KNN RMSE:", sqrt(mse)))
print(paste("KNN MAE:", mae))
print(paste("KNN MAPE:", mape))
print(paste("KNN RMSLE:", rmsle))
print(paste("KNN R-squared:", round(r2, 4)))
print(paste("KNN Adjusted R-squared:", round(adj.r2, 4)))

mse.scores <- NULL

# Elbow method
for(i in 1:21) {
  knn.model <- knn.reg(x_train,x_test,y_train,k=i)
  knn.predictions <- knn.model$pred
  mse <- mean((test$Yield_tons_per_hectare - knn.predictions) ** 2)
  mse.scores[i] <- mse
}

elbow.df <- data.frame(k=1:21,mse=mse.scores)
ggplotly(ggplot(data=elbow.df,mapping=aes(x=k,y=mse)) + geom_line(alpha=0.5) + theme_bw())

# Based on the elbow method, the most optimal value of k is 3.

best.knn.model <- knn.reg(x_train,x_test,y_train,k=3,algorithm='kd_tree')
print(summary(best.knn.model))

knn.predictions <- best.knn.model$pred

# MSE
mse <- mean((test$Yield_tons_per_hectare - knn.predictions) ** 2)

# RMSE
rmse <- sqrt(mse)

# MAE
mae <- mean(abs(test$Yield_tons_per_hectare - knn.predictions))

# MAPE
mape <- mean(abs((test$Yield_tons_per_hectare - knn.predictions) / test$Yield_tons_per_hectare)) * 100

# RMSLE
rmsle <- sqrt(mean((log1p(knn.predictions) - log1p(test$Yield_tons_per_hectare))^2))

# R-squared
sst <- sum((test$Yield_tons_per_hectare - mean(test$Yield_tons_per_hectare)) ** 2) # Sum of Squares Total (SST)
sse <- sum((test$Yield_tons_per_hectare - knn.predictions) ** 2) # Sum of Squares Error (SSE)

r2 <- 1 - sse / sst

# Adjusted R-squared
n <- nrow(test)       # Number of observations
p <- length(knn.model) - 1  # Number of predictors (excluding intercept)
adj.r2 <- 1 - (1 - r2) * ((n - 1) / (n - p - 1))

print(paste("KNN MSE:", mse))
print(paste("KNN RMSE:", sqrt(mse)))
print(paste("KNN MAE:", mae))
print(paste("KNN MAPE:", mape))
print(paste("KNN RMSLE:", rmsle))
print(paste("KNN R-squared:", round(r2, 4)))
print(paste("KNN Adjusted R-squared:", round(adj.r2, 4)))

model.performance <- rbind(model.performance,data.frame(model="K Nearest Neighbors",r2.score=r2))

# Decision Tree
dt.model <- rpart(formula=Yield_tons_per_hectare ~ ., data=train, method='anova')
print(summary(dt.model))
# plot(dt.model,uniform=T,main='Decision Tree')
# text(dt.model,use.n=T,all=T)

printcp(dt.model)

# Plot decision tree model
prp(dt.model)

dt.predictions <- predict(dt.model,newdata=test)

# MSE
mse <- mean((test$Yield_tons_per_hectare - dt.predictions) ** 2)

# RMSE
rmse <- sqrt(mse)

# MAE
mae <- mean(abs(test$Yield_tons_per_hectare - dt.predictions))

# MAPE
mape <- mean(abs((test$Yield_tons_per_hectare - dt.predictions) / test$Yield_tons_per_hectare)) * 100

# RMSLE
rmsle <- sqrt(mean((log1p(dt.predictions) - log1p(test$Yield_tons_per_hectare))^2))

# R-squared
sst <- sum((test$Yield_tons_per_hectare - mean(test$Yield_tons_per_hectare)) ** 2) # Sum of Squares Total (SST)
sse <- sum((test$Yield_tons_per_hectare - dt.predictions) ** 2) # Sum of Squares Error (SSE)

r2 <- 1 - sse / sst

# Adjusted R-squared
n <- nrow(test)       # Number of observations
p <- length(dt.model) - 1  # Number of predictors (excluding intercept)
adj.r2 <- 1 - (1 - r2) * ((n - 1) / (n - p - 1))

print(paste("Decision Tree MSE:", mse))
print(paste("Decision Tree RMSE:", sqrt(mse)))
print(paste("Decision Tree MAE:", mae))
print(paste("Decision Tree MAPE:", mape))
print(paste("Decision Tree RMSLE:", rmsle))
print(paste("Decision Tree R-squared:", round(r2, 4)))
print(paste("Decision Tree Adjusted R-squared:", round(adj.r2, 4)))

model.performance <- rbind(model.performance,data.frame(model="Decision Tree",r2.score=r2))

# Random Forest model
rf.model <- randomForest(formula=Yield_tons_per_hectare ~ ., data=train, ntree=400, maxnodes=30)
print(summary(rf.model))

rf.predictions <- predict(rf.model,newdata=test)

# MSE
mse <- mean((test$Yield_tons_per_hectare - rf.predictions) ** 2)

# RMSE
rmse <- sqrt(mse)

# MAE
mae <- mean(abs(test$Yield_tons_per_hectare - rf.predictions))

# MAPE
mape <- mean(abs((test$Yield_tons_per_hectare - rf.predictions) / test$Yield_tons_per_hectare)) * 100

# RMSLE
rmsle <- sqrt(mean((log1p(rf.predictions) - log1p(test$Yield_tons_per_hectare))^2))

# R-squared
sst <- sum((test$Yield_tons_per_hectare - mean(test$Yield_tons_per_hectare)) ** 2) # Sum of Squares Total (SST)
sse <- sum((test$Yield_tons_per_hectare - rf.predictions) ** 2) # Sum of Squares Error (SSE)

r2 <- 1 - sse / sst

# Adjusted R-squared
n <- nrow(test)       # Number of observations
p <- length(rf.model) - 1  # Number of predictors (excluding intercept)
adj.r2 <- 1 - (1 - r2) * ((n - 1) / (n - p - 1))

print(paste("Random Forest MSE:", mse))
print(paste("Random Forest RMSE:", sqrt(mse)))
print(paste("Random Forest MAE:", mae))
print(paste("Random Forest MAPE:", mape))
print(paste("Random Forest RMSLE:", rmsle))
print(paste("Random Forest R-squared:", round(r2, 4)))
print(paste("Random Forest Adjusted R-squared:", round(adj.r2, 4)))

model.performance <- rbind(model.performance,data.frame(model="Random Forest",r2.score=r2))

# XGBoost model

# Separate features and target
x_train <- train[, setdiff(names(train), "Yield_tons_per_hectare")]
x_train_matrix <- model.matrix(Yield_tons_per_hectare ~ . - 1,data=train)
y_train <- train$Yield_tons_per_hectare
x_test <- test[,setdiff(names(test), "Yield_tons_per_hectare")]
x_test_matrix <- model.matrix(Yield_tons_per_hectare ~ . -1, data=test)
y_test <- test$Yield_tons_per_hectare

# Create DMatrix
dmatrix.train <- xgb.DMatrix(data=x_train_matrix,label=train$Yield_tons_per_hectare)
dmatrix.test <- xgb.DMatrix(data=x_test_matrix,label=test$Yield_tons_per_hectare)

xgb.model <- xgboost(data=dmatrix.train,max_depth=4,eta=0.1,nrounds=100,objective='reg:squarederror',max_depth=8,booster='gbtree',verbose=0)
print(summary(xgb.model))

xgb.predictions <- predict(xgb.model,newdata=dmatrix.test)

# MSE
mse <- mean((test$Yield_tons_per_hectare != xgb.predictions) ** 2)
# RMSE
rmse <- sqrt(mse)
# MAE
mae <- mean(abs(test$Yield_tons_per_hectare != xgb.predictions))
# MAPE
mape <- mean(abs((test$Yield_tons_per_hectare - xgb.predictions) / test$Yield_tons_per_hectare)) * 100
# RMSLE
rmsle <- sqrt(mean(log1p(xgb.predictions) - log1p(test$Yield_tons_per_hectare)) ** 2)

# R2
sst <- sum((test$Yield_tons_per_hectare - mean(test$Yield_tons_per_hectare)) ** 2) # Sum of Squares Total
sse <- sum((test$Yield_tons_per_hectare - xgb.predictions) ** 2) # Sum of Squares Error

r2 <- 1 - sse / sst

# Adjusted R-squared
n <- nrow(test)       # Number of observations
p <- length(rf.model) - 1  # Number of predictors (excluding intercept)
adj.r2 <- 1 - (1 - r2) * ((n - 1) / (n - p - 1))

cat("XGB MSE:", mse, "\n")
cat("XGB RMSE:", rmse, "\n")
cat("XGB MAE:", mae, "\n")
cat("XGB MAPE:", mape, "\n")
cat("XGB RMSLE:", rmsle, "\n")
cat("XGB R-squared:", r2, "\n")
cat("XGB Adj R-squared:", adj.r2, "\n")

model.performance <- rbind(model.performance,data.frame(model="XGBoost",r2.score=r2))

# Bayesian Regression
br.model <- brm(formula=Yield_tons_per_hectare ~ ., data=train,seed=101,silent=0,backend='cmdstanr',algorithm='sampling',cores=3,iter=1000,chains=4)
print(summary(br.model))

br.predictions <- predict(br.model,newdata=test)

# MSE
mse <- mean((test$Yield_tons_per_hectare != br.predictions) ** 2)
# RMSE
rmse <- sqrt(mse)
# MAE
mae <- mean(abs(test$Yield_tons_per_hectare != br.predictions))
# MAPE
mape <- mean(abs((test$Yield_tons_per_hectare - br.predictions) / test$Yield_tons_per_hectare)) * 100
# RMSLE
rmsle <- sqrt(mean(log1p(br.predictions) - log1p(test$Yield_tons_per_hectare)) ** 2)

# R2
sst <- sum((test$Yield_tons_per_hectare - mean(test$Yield_tons_per_hectare)) ** 2) # Sum of Squares Total
sse <- sum((test$Yield_tons_per_hectare - br.predictions) ** 2) # Sum of Squares Error

r2 <- 1 - sse / sst

# Adjusted R-squared
n <- nrow(test)       # Number of observations
p <- length(br.model) - 1  # Number of predictors (excluding intercept)
adj.r2 <- 1 - (1 - r2) * ((n - 1) / (n - p - 1))

cat("Bayesian Regression MSE:", mse, "\n")
cat("Bayesian Regression RMSE:", rmse, "\n")
cat("Bayesian Regression MAE:", mae, "\n")
cat("Bayesian Regression MAPE:", mape, "\n")
cat("Bayesian Regression RMSLE:", rmsle, "\n")
cat("Bayesian Regression R-squared:", r2, "\n")
cat("Bayesian Regression Adj R-squared:", adj.r2, "\n")

model.performance <- rbind(model.performance,data.frame(model="Bayesian Regression",r2.score=r2))

# Compute min and max from training set
mins <- apply(x_train_matrix, 2, min)
maxs <- apply(x_train_matrix, 2, max)
ranges <- maxs - mins

# Avoid division by zero (in case of constant columns)
ranges[ranges == 0] <- 1

# Scale train and test sets
scaled.x.train <- as.data.frame(scale(x_train_matrix, center = mins, scale = ranges))
scaled.x.test  <- as.data.frame(scale(x_test_matrix,  center = mins, scale = ranges))

# Combine scaled predictions and target variable into a dataframe
scaled.train.df <- cbind(scaled.x.train,y_train)
scaled.test.df <- cbind(scaled.x.test,y_test)

scaled.train.df <- scaled.train.df %>% rename(Yield_tons_per_hectare=y_train)
scaled.test.df <- scaled.test.df %>% rename(Yield_tons_per_hectare=y_test)

# Neural network
nn.model <- nnet(formula=Yield_tons_per_hectare ~ .,data=scaled.train.df,size=5,decay=1e-2,linout=TRUE,skip=FALSE)
plotnet(nn.model) # Plot the neural network
garson(nn.model) # Visualize feature importance
olden(nn.model) # Visualize individual connection strengths
print(summary(nn.model))

nn.predictions <- predict(nn.model,newdata=scaled.test.df[,setdiff(names(scaled.test.df),"Yield_tons_per_hectare")])

# MSE
mse <- mean((scaled.test.df$Yield_tons_per_hectare - nn.predictions) ** 2)
# RMSE
rmse <- sqrt(mse)
# MAE
mae <- mean(abs(scaled.test.df$Yield_tons_per_hectare - nn.predictions))
# MAPE
mape <- mean(abs((scaled.test.df$Yield_tons_per_hectare - nn.predictions) / scaled.test.df$Yield_tons_per_hectare)) * 100
# RMSLE
rmsle <- sqrt(mean(log1p(scaled.test.df$Yield_tons_per_hectare) - log1p(nn.predictions)) ** 2)
# R2
sse <- sum((scaled.test.df$Yield_tons_per_hectare - nn.predictions) ** 2)
sst <- sum((scaled.test.df$Yield_tons_per_hectare - mean(scaled.test.df$Yield_tons_per_hectare)) ** 2)

r2 <- 1 - sse / sst

# Adjusted R-squared
n <- nrow(scaled.test.df)  # Number of observations
p <- length(nn.model) - 1  # Number of predictors (excluding intercept)
adj.r2 <- 1 - (1 - r2) * ((n - 1) / (n - p - 1))

cat("Neural Network MSE:", mse, "\n")
cat("Neural Network RMSE:", rmse, "\n")
cat("Neural Network MAE:", mae, "\n")
cat("Neural Network MAPE:", mape, "\n")
cat("Neural Network RMSLE:", rmsle, "\n")
cat("Neural Network R-squared:", r2, "\n")
cat("Neural Network Adj R-squared:", adj.r2, "\n")

model.performance <- rbind(model.performance,data.frame(model="Neural Network",r2.score=r2))

# Light Gradient Boosting Machine (LGBM)
lgb.train.data <- lgb.Dataset(data=x_train_matrix,label=y_train)
lgb.test.data <- lgb.Dataset(data=x_test_matrix,label=y_test)

lgbm.model <- lightgbm(data=lgb.train.data,nrounds=100,objective='regression',verbose=0)
print(summary(lgbm.model))

lgbm.predictions <- predict(lgbm.model,newdata=x_test_matrix)

mse <- mean((y_test != lgbm.predictions) ** 2) # MSE
rmse <- sqrt(mse) # RMSE
mae <- mean(abs((y_test - lgbm.predictions))) # MAE
mape <- mean(abs((y_test - lgbm.predictions) / y_test)) * 100 # MAPE
rmsle <- sqrt(mean((log1p(y_test) != log1p(lgbm.predictions)) ** 2)) # RMSLE

# R2
sst <- sum((y_test - mean(y_test)) ** 2)
sse <- sum((y_test - lgbm.predictions) ** 2)

r2 <- 1 - sse / sst

# Adjusted R2
n <- nrow(x_test_matrix)       # Number of observations
p <- length(lgbm.model) - 1  # Number of predictors (excluding intercept)
adj.r2 <- 1 - (1 - r2) * ((n - 1) / (n - p - 1))

cat("Light GBM MSE:", mse, "\n")
cat("Light GBM RMSE:", rmse, "\n")
cat("Light GBM MAE:", mae, "\n")
cat("Light GBM MAPE:", mape, "\n")
cat("Light GBM RMSLE:", rmsle, "\n")
cat("Light GBM R-squared:", r2, "\n")
cat("Light GBM Adj R-squared:", adj.r2, "\n")        

model.performance <- rbind(model.performance,data.frame(model="Light GBM",r2.score=r2))

# CatBoost
catboost <- import("catboost")
np <- import("numpy")

train.pool <- catboost$Pool(data=np$array(x_train_matrix),label=y_train)
test.pool <- catboost$Pool(data=np$array(x_test_matrix),label=y_test)

cat.model <- catboost$CatBoostRegressor(iterations=100,learning_rate=0.1,depth=6,loss_function='RMSE')
cat.model$fit(train.pool)

cat.predictions <- cat.model$predict(test.pool)

mse <- mean((y_test != cat.predictions) ** 2) # MSE
rmse <- sqrt(mse) # RMSE
mae <- mean(abs((y_test - cat.predictions))) # MAE
mape <- mean(abs((y_test - cat.predictions) / y_test)) * 100 # MAPE
rmsle <- sqrt(mean((log1p(y_test) != log1p(cat.predictions)) ** 2)) # RMSLE

# R2
sst <- sum((y_test - mean(y_test)) ** 2)
sse <- sum((y_test - cat.predictions) ** 2)

r2 <- 1 - sse / sst

# Adjusted R2
n <- nrow(x_test_matrix)       # Number of observations
p <- length(cat.model) - 1  # Number of predictors (excluding intercept)
adj.r2 <- 1 - (1 - r2) * ((n - 1) / (n - p - 1))

cat("CatBoost MSE:", mse, "\n")
cat("CatBoost RMSE:", rmse, "\n")
cat("CatBoost MAE:", mae, "\n")
cat("CatBoost MAPE:", mape, "\n")
cat("CatBoost RMSLE:", rmsle, "\n")
cat("CatBoost R-squared:", r2, "\n")
cat("CatBoost Adj R-squared:", adj.r2, "\n")    

model.performance <- rbind(model.performance,data.frame(model="CatBoost",r2.score=r2))

# Quantile regression
qr.model <- rq(formula=Yield_tons_per_hectare ~ .,data=train,tau=0.5)
print(summary(qr.model))

qr.predictions <- predict(qr.model,newdata=test)

mse <- mean((y_test != qr.predictions) ** 2) # MSE
rmse <- sqrt(mse) # RMSE
mae <- mean(abs((y_test - qr.predictions))) # MAE
mape <- mean(abs((y_test - qr.predictions) / y_test)) * 100 # MAPE
rmsle <- sqrt(mean((log1p(y_test) != log1p(qr.predictions)) ** 2)) # RMSLE

# R2
sst <- sum((y_test - mean(y_test)) ** 2)
sse <- sum((y_test - qr.predictions) ** 2)

r2 <- 1 - sse / sst

# Adjusted R2
n <- nrow(test)       # Number of observations
p <- length(qr.model) - 1  # Number of predictors (excluding intercept)
adj.r2 <- 1 - (1 - r2) * ((n - 1) / (n - p - 1))

cat("Quantile Regression MSE:", mse, "\n")
cat("Quantile Regression RMSE:", rmse, "\n")
cat("Quantile Regression MAE:", mae, "\n")
cat("Quantile Regression MAPE:", mape, "\n")
cat("Quantile Regression RMSLE:", rmsle, "\n")
cat("Quantile Regression R-squared:", r2, "\n")
cat("Quantile Regression Adj R-squared:", adj.r2, "\n") 

model.performance <- rbind(model.performance,data.frame(model="Quantile Regression",r2.score=r2))

# Generalized Additive Model
gam.model <- gam(Yield_tons_per_hectare ~ s(Rainfall_mm) + s(Temperature_Celsius) +
                   s(Days_to_Harvest) + Soil_Type + Crop + Fertilizer_Used + Irrigation_Used + Region,
                 data=train)
print(summary(gam.model))

gam.predictions <- predict(gam.model, newdata=test)

mse <- mean((y_test != gam.predictions) ** 2) # MSE
rmse <- sqrt(mse) # RMSE
mae <- mean(abs((y_test - gam.predictions))) # MAE
mape <- mean(abs((y_test - gam.predictions) / y_test)) * 100 # MAPE
rmsle <- sqrt(mean((log1p(y_test) != log1p(gam.predictions)) ** 2)) # RMSLE

# R2
sst <- sum((y_test - mean(y_test)) ** 2)
sse <- sum((y_test - gam.predictions) ** 2)

r2 <- 1 - sse / sst

# Adjusted R2
n <- nrow(test)       # Number of observations
p <- length(gam.model) - 1  # Number of predictors (excluding intercept)
adj.r2 <- 1 - (1 - r2) * ((n - 1) / (n - p - 1))

cat("Generalized Additive Model MSE:", mse, "\n")
cat("Generalized Additive Model RMSE:", rmse, "\n")
cat("Generalized Additive Model MAE:", mae, "\n")
cat("Generalized Additive Model MAPE:", mape, "\n")
cat("Generalized Additive Model RMSLE:", rmsle, "\n")
cat("Generalized Additive Model R-squared:", r2, "\n")
cat("Generalized Additive Model Adj R-squared:", adj.r2, "\n") 

model.performance <- rbind(model.performance,data.frame(model="Generalized Additive Model",r2.score=r2))

# Hyperparameter Tuning & Cross Validation

# Random Forest
tuned.rf <- tuneRF(x=train[,-which(names(train)=="Yield_tons_per_hectare")],
                   y=train$Yield_tons_per_hectare,
                   ntreeTry=100,
                   stepFactor=1.5,
                   improve=0.05,
                   trace=TRUE,
                   doBest=TRUE,
                   maxnodes=20)

tuned.rf.predictions <- predict(tuned.rf,newdata=test)

mse <- mean((y_test != tuned.rf.predictions) ** 2) # MSE
rmse <- sqrt(mse) # RMSE
mae <- mean(abs((y_test - tuned.rf.predictions))) # MAE
mape <- mean(abs((y_test - tuned.rf.predictions) / y_test)) * 100 # MAPE
rmsle <- sqrt(mean((log1p(y_test) != log1p(tuned.rf.predictions)) ** 2)) # RMSLE

# R2
sst <- sum((y_test - mean(y_test)) ** 2)
sse <- sum((y_test - tuned.rf.predictions) ** 2)

r2 <- 1 - sse / sst

# Adjusted R2
n <- nrow(test)       # Number of observations
p <- length(tuned.rf) - 1  # Number of predictors (excluding intercept)
adj.r2 <- 1 - (1 - r2) * ((n - 1) / (n - p - 1))

cat("Tuned Random Forest MSE:", mse, "\n")
cat("Tuned Random Forest RMSE:", rmse, "\n")
cat("Tuned Random Forest MAE:", mae, "\n")
cat("Tuned Random Forest MAPE:", mape, "\n")
cat("Tuned Random Forest RMSLE:", rmsle, "\n")
cat("Tuned Random Forest R-squared:", r2, "\n")
cat("Tuned Random Forest Adj R-squared:", adj.r2, "\n") 

model.performance <- rbind(model.performance, data.frame(model='Tuned Random Forest', r2.score=r2))

# XGBoost
tune.grid <- expand.grid(nrounds=c(100,200),
                         max_depth=c(4,6,8),
                         eta=c(0.01,0.1),
                         gamma=0,
                         colsample_bytree=1,
                         min_child_weight=1,
                         subsample=1)

ctrl <- trainControl(method="cv",number=5)

xgb.tuned <- train(x=x_train_matrix, 
                   y=y_train,
                   method="xgbTree",
                   trControl=ctrl,
                   tuneGrid=tune.grid)

tuned.xgb.predictions <- predict(xgb.tuned,newdata=x_test_matrix)

mse <- mean((y_test != tuned.xgb.predictions) ** 2) # MSE
rmse <- sqrt(mse) # RMSE
mae <- mean(abs((y_test - tuned.xgb.predictions))) # MAE
mape <- mean(abs((y_test - tuned.xgb.predictions) / y_test)) * 100 # MAPE
rmsle <- sqrt(mean((log1p(y_test) != log1p(tuned.xgb.predictions)) ** 2)) # RMSLE

# R2
sst <- sum((y_test - mean(y_test)) ** 2)
sse <- sum((y_test - tuned.xgb.predictions) ** 2)

r2 <- 1 - sse / sst

# Adjusted R2
n <- nrow(test)       # Number of observations
p <- length(xgb.tuned) - 1  # Number of predictors (excluding intercept)
adj.r2 <- 1 - (1 - r2) * ((n - 1) / (n - p - 1))

cat("Tuned XGBoost MSE:", mse, "\n")
cat("Tuned XGBoost RMSE:", rmse, "\n")
cat("Tuned XGBoost MAE:", mae, "\n")
cat("Tuned XGBoost MAPE:", mape, "\n")
cat("Tuned XGBoost RMSLE:", rmsle, "\n")
cat("Tuned XGBoost R-squared:", r2, "\n")
cat("Tuned XGBoost Adj R-squared:", adj.r2, "\n") 

model.performance <- rbind(model.performance, data.frame(model='Tuned XGBoost', r2.score=r2))

# Light GBM
params <- list(objective="regression",
               metric="rmse",
               num_leaves=31,
               learning_rate=0.1)

lgb.cv <- lgb.cv(params=params,
                 data=lgb.train.data,
                 nrounds=100,
                 nfold=5,
                 verbose=0,
                 early_stopping_rounds=10)

best_iter <- lgb.cv$best_iter
final.lgb <- lgb.train(params=params,
                       data=lgb.train.data,
                       nrounds=best_iter)

tuned.lgb.predictions <- predict(final.lgb,newdata=x_test_matrix)

mse <- mean((y_test != tuned.lgb.predictions) ** 2) # MSE
rmse <- sqrt(mse) # RMSE
mae <- mean(abs((y_test - tuned.lgb.predictions))) # MAE
mape <- mean(abs((y_test - tuned.lgb.predictions) / y_test)) * 100 # MAPE
rmsle <- sqrt(mean((log1p(y_test) != log1p(tuned.lgb.predictions)) ** 2)) # RMSLE

# R2
sst <- sum((y_test - mean(y_test)) ** 2)
sse <- sum((y_test - tuned.lgb.predictions) ** 2)

r2 <- 1 - sse / sst

# Adjusted R2
n <- nrow(test)       # Number of observations
p <- length(final.lgb) - 1  # Number of predictors (excluding intercept)
adj.r2 <- 1 - (1 - r2) * ((n - 1) / (n - p - 1))

cat("Tuned Light GBM MSE:", mse, "\n")
cat("Tuned Light GBM RMSE:", rmse, "\n")
cat("Tuned Light GBM MAE:", mae, "\n")
cat("Tuned Light GBM MAPE:", mape, "\n")
cat("Tuned Light GBM RMSLE:", rmsle, "\n")
cat("Tuned Light GBM R-squared:", r2, "\n")
cat("Tuned Light GBM Adj R-squared:", adj.r2, "\n") 

model.performance <- rbind(model.performance, data.frame(model='Tuned Light GBM', r2.score=r2))

# Neural Network
nnet.grid <- expand.grid(size=c(3, 5, 7),
                         decay=c(0.01, 0.001))

ctrl <- trainControl(method="cv",number=5)

nnet.tuned <- train(Yield_tons_per_hectare ~ .,
                    data=scaled.train.df,
                    method="nnet",
                    tuneGrid=nnet.grid,
                    trControl=ctrl,
                    linout=TRUE,
                    trace=FALSE)

tuned.nn.predictions <- predict(nnet.tuned,newdata=scaled.test.df)
mse <- mean((y_test != tuned.nn.predictions) ** 2) # MSE
rmse <- sqrt(mse) # RMSE
mae <- mean(abs((y_test - tuned.nn.predictions))) # MAE
mape <- mean(abs((y_test - tuned.nn.predictions) / y_test)) * 100 # MAPE
rmsle <- sqrt(mean((log1p(y_test) != log1p(tuned.nn.predictions)) ** 2)) # RMSLE

# R2
sst <- sum((y_test - mean(y_test)) ** 2)
sse <- sum((y_test - tuned.nn.predictions) ** 2)

r2 <- 1 - sse / sst

# Adjusted R2
n <- nrow(scaled.test.df)       # Number of observations
p <- length(nnet.tuned) - 1  # Number of predictors (excluding intercept)
adj.r2 <- 1 - (1 - r2) * ((n - 1) / (n - p - 1))

cat("Tuned Neural Network MSE:", mse, "\n")
cat("Tuned Neural Network RMSE:", rmse, "\n")
cat("Tuned Neural Network MAE:", mae, "\n")
cat("Tuned Neural Network MAPE:", mape, "\n")
cat("Tuned Neural Network RMSLE:", rmsle, "\n")
cat("Tuned Neural Network R-squared:", r2, "\n")
cat("Tuned Neural Network Adj R-squared:", adj.r2, "\n") 

model.performance <- rbind(model.performance, data.frame(model='Tuned Neural Network', r2.score=r2))

# CatBoost
params <- list(iterations=c(100,200),
               learning_rate=c(0.01,0.1),
               depth=c(4,6,8))

best.rmse <- Inf

for (iter in params$iterations) {
  for (lr in params$learning_rate) {
    for (d in params$depth) {
      model <- catboost$CatBoostRegressor(iterations=iter,learning_rate=lr,depth=d,loss_function='RMSE')
      model$fit(train.pool,eval_set=test.pool,verbose=FALSE)
      pred <- model$predict(test.pool)
      rmse <- sqrt(mean((y_test - pred)^2))
      if (rmse < best.rmse) {
        best.cat.model <- model
        best.rmse <- rmse
      }
    }
  }
}

tuned.cat.predictions <- best.cat.model$predict(test.pool)
mse <- mean((y_test != tuned.cat.predictions) ** 2) # MSE
rmse <- sqrt(mse) # RMSE
mae <- mean(abs((y_test - tuned.cat.predictions))) # MAE
mape <- mean(abs((y_test - tuned.cat.predictions) / y_test)) * 100 # MAPE
rmsle <- sqrt(mean((log1p(y_test) != log1p(tuned.cat.predictions)) ** 2)) # RMSLE

# R2
sst <- sum((y_test - mean(y_test)) ** 2)
sse <- sum((y_test - tuned.cat.predictions) ** 2)

r2 <- 1 - sse / sst

# Adjusted R2
n <- nrow(x_test_matrix)     # Number of observations
p <- length(cat.model$feature_names_)   # Number of predictors (excluding intercept)
adj.r2 <- 1 - (1 - r2) * ((n - 1) / (n - p - 1))

cat("Tuned CatBoost MSE:", mse, "\n")
cat("Tuned CatBoost RMSE:", rmse, "\n")
cat("Tuned CatBoost MAE:", mae, "\n")
cat("Tuned CatBoost MAPE:", mape, "\n")
cat("Tuned CatBoost RMSLE:", rmsle, "\n")
cat("Tuned CatBoost R-squared:", r2, "\n")
cat("Tuned CatBoost Adj R-squared:", adj.r2, "\n") 

model.performance <- rbind(model.performance,data.frame(model='Tuned CatBoost', r2.score=r2))

# Decision Tree
ctrl <- trainControl(method="cv",number=5)

grid <- expand.grid(cp=seq(0.001,0.05,by=0.005))

tuned.dt.model <- train(Yield_tons_per_hectare ~ ., 
                    data=train,
                    method="rpart",
                    trControl=ctrl,
                    tuneGrid=grid)
print(tuned.dt.model)
plot(tuned.dt.model)

tuned.dt.predictions <- predict(tuned.dt.model,newdata=test)
mse <- mean((y_test != tuned.dt.predictions) ** 2) # MSE
rmse <- sqrt(mse) # RMSE
mae <- mean(abs((y_test - tuned.dt.predictions))) # MAE
mape <- mean(abs((y_test - tuned.dt.predictions) / y_test)) * 100 # MAPE
rmsle <- sqrt(mean((log1p(y_test) != log1p(tuned.dt.predictions)) ** 2)) # RMSLE

# R2
sst <- sum((y_test - mean(y_test)) ** 2)
sse <- sum((y_test - tuned.dt.predictions) ** 2)

r2 <- 1 - sse / sst

# Adjusted R2
n <- nrow(test)     # Number of observations
p <- length(tuned.dt.model) - 1  # Number of predictors (excluding intercept)
adj.r2 <- 1 - (1 - r2) * ((n - 1) / (n - p - 1))

cat("Tuned Decision Tree MSE:", mse, "\n")
cat("Tuned Decision Tree RMSE:", rmse, "\n")
cat("Tuned Decision Tree MAE:", mae, "\n")
cat("Tuned Decision Tree MAPE:", mape, "\n")
cat("Tuned Decision Tree RMSLE:", rmsle, "\n")
cat("Tuned Decision Tree R-squared:", r2, "\n")
cat("Tuned Decision Tree Adj R-squared:", adj.r2, "\n") 

model.performance <- rbind(model.performance,data.frame(model='Tuned Decision Tree', r2.score=r2))

# Quantile Regression
taus <- seq(0.1, 0.9, 0.1)

qr.models <- lapply(taus, function(tau) {
  model <- rq(Yield_tons_per_hectare ~ ., data=train,tau=tau)
  preds <- predict(model,newdata=test)
  rmse <- sqrt(mean((test$Yield_tons_per_hectare - preds)^2))
  
  return(list(model=model,tau=tau,RMSE=rmse))
})

# Create a summary dataframe
qr.results <- do.call(rbind, lapply(qr.models, function(res) {
  data.frame(tau=res$tau,RMSE=res$RMSE)
}))

best.idx <- which.min(qr.results$RMSE)
best.qr.model <- qr.models[[best.idx]]$model

tuned.qr.predictions <- predict(best.qr.model,newdata=test)
mse <- mean((y_test != tuned.qr.predictions) ** 2) # MSE
rmse <- sqrt(mse) # RMSE
mae <- mean(abs((y_test - tuned.qr.predictions))) # MAE
mape <- mean(abs((y_test - tuned.qr.predictions) / y_test)) * 100 # MAPE
rmsle <- sqrt(mean((log1p(y_test) != log1p(tuned.qr.predictions)) ** 2)) # RMSLE

# R2
sst <- sum((y_test - mean(y_test)) ** 2)
sse <- sum((y_test - tuned.qr.predictions) ** 2)

r2 <- 1 - sse / sst

# Adjusted R2
n <- nrow(test)     # Number of observations
p <- length(best.qr.model) - 1  # Number of predictors (excluding intercept)
adj.r2 <- 1 - (1 - r2) * ((n - 1) / (n - p - 1))

cat("Tuned Quantile Regression MSE:", mse, "\n")
cat("Tuned Quantile Regression RMSE:", rmse, "\n")
cat("Tuned Quantile Regression MAE:", mae, "\n")
cat("Tuned Quantile Regression MAPE:", mape, "\n")
cat("Tuned Quantile Regression RMSLE:", rmsle, "\n")
cat("Tuned Quantile Regression R-squared:", r2, "\n")
cat("Tuned Quantile Regression Adj R-squared:", adj.r2, "\n") 

model.performance <- rbind(model.performance, data.frame(model="Tuned Quantile Regression", r2.score=r2))

# Generalized Additive Models (GAM)
gam.model <- gam(Yield_tons_per_hectare ~ s(Rainfall_mm, k = 6) + s(Temperature_Celsius, k = 6) +
                   s(Days_to_Harvest, k = 6) + Soil_Type + Crop + Fertilizer_Used +
                   Irrigation_Used + Region,data=train)

summary(gam.model)
gam.check(gam.model)

tuned.gam.predictions <- predict(gam.model,newdata=test)
mse <- mean((y_test != tuned.gam.predictions) ** 2) # MSE
rmse <- sqrt(mse) # RMSE
mae <- mean(abs((y_test - tuned.gam.predictions))) # MAE
mape <- mean(abs((y_test - tuned.gam.predictions) / y_test)) * 100 # MAPE
rmsle <- sqrt(mean((log1p(y_test) != log1p(tuned.gam.predictions)) ** 2)) # RMSLE

# R2
sst <- sum((y_test - mean(y_test)) ** 2)
sse <- sum((y_test - tuned.gam.predictions) ** 2)

r2 <- 1 - sse / sst

# Adjusted R2
n <- nrow(test)     # Number of observations
p <- length(gam.model) - 1  # Number of predictors (excluding intercept)
adj.r2 <- 1 - (1 - r2) * ((n - 1) / (n - p - 1))

cat("Tuned GAM MSE:", mse, "\n")
cat("Tuned GAM RMSE:", rmse, "\n")
cat("Tuned GAM MAE:", mae, "\n")
cat("Tuned GAM MAPE:", mape, "\n")
cat("Tuned GAM RMSLE:", rmsle, "\n")
cat("Tuned GAM R-squared:", r2, "\n")
cat("Tuned GAM Adj R-squared:", adj.r2, "\n") 

model.performance <- rbind(model.performance,data.frame(model="Tuned GAM",r2.score=r2))

# Compare all models and save the best one
print(model.performance %>% arrange(desc(r2.score)))
best.model <- model.performance[which.max(model.performance$r2.score),]
print(paste("Best model is:", best.model$model, "with r2 score:", best.model$r2.score))

# Based on the final results, we can clearly infer that Linear Regression turned out to be the best performing model achieving more than 91% R2 score and Adjusted R2 score on the test set.

# Saving the best performing model
saveRDS(lr.model, "crop_yield_predictor.rds")

# Load the saved model
loaded.model <- readRDS('crop_yield_predictor.rds')

loaded.model.predictions <- predict(loaded.model,newdata=test)

mse <- mean((y_test != loaded.model.predictions) ** 2) # MSE
rmse <- sqrt(mse) # RMSE
mae <- mean(abs((y_test - loaded.model.predictions))) # MAE
mape <- mean(abs((y_test - loaded.model.predictions) / y_test)) * 100 # MAPE
rmsle <- sqrt(mean((log1p(y_test) != log1p(loaded.model.predictions)) ** 2)) # RMSLE

# R2
sst <- sum((y_test - mean(y_test)) ** 2)
sse <- sum((y_test - loaded.model.predictions) ** 2)

r2 <- 1 - sse / sst

# Adjusted R2
n <- nrow(test)     # Number of observations
p <- length(loaded.model) - 1  # Number of predictors (excluding intercept)
adj.r2 <- 1 - (1 - r2) * ((n - 1) / (n - p - 1))

cat("Best Model (LR) MSE:", mse, "\n")
cat("Best Model (LR) RMSE:", rmse, "\n")
cat("Best Model (LR) MAE:", mae, "\n")
cat("Best Model (LR) MAPE:", mape, "\n")
cat("Best Model (LR) RMSLE:", rmsle, "\n")
cat("Best Model (LR) R-squared:", r2, "\n")
cat("Best Model (LR) Adj R-squared:", adj.r2, "\n") 