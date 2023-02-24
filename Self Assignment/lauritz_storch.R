# ------------------------------------------------------------------------------
# self-study
# course: Data Analytics I: Predictive Econometrics
# author: Lauritz Storch (21-607-015)
# ------------------------------------------------------------------------------

# Clear R
rm(list=ls())
cat("\014")
graphics.off()

# Set Seed
set.seed(2023)

## Set working directory to load data
setwd("~/Desktop/Self Assignment")


## Packages and Library
# Packages function
install_if_missing <- function(p) {
  if (p %in% rownames(installed.packages()) == F) {
    try(install.packages(p, dependencies = T))
  } else {
    cat(paste("Skipping already installed package:", p, "\n"))
  }
}

# Define packages
# packages = c("gridExtra","dplyr","kableExtra","knitr","lmtest","glmnet","corrplot",
#              "gglasso","caret","stargazer") # R-Markdown
packages = c("gridExtra","dplyr","lmtest","glmnet","corrplot",
             "gglasso","caret","stargazer") # Code

# Install
invisible(sapply(packages, install_if_missing))

# read library
for(pkg in packages){
  library(pkg, character.only = TRUE)
}


## Read in data
load("obesity.Rdata")
load("obesity_predict.Rdata")


## Create folder for prediction results (might differ in a Windows environment)
dir.create(paste0(getwd(),"//Prediction Results"))


# Plot histogram
discrete <- ggplot(data = obesity, aes(x = BMI)) +
  geom_histogram(bins=30)
continuous <- ggplot(data = obesity, aes(x = BMI)) +
  stat_density()
grid.arrange(arrangeGrob(discrete, continuous, ncol=2, nrow=1), heights=c(2,1))


# Print summary for variable Age
summary_age <- cbind(summary(obesity$Age),summary(obesity_predict$Age))
colnames(summary_age) <- c("Age train","Age test")
summary_age


# Data Processing --------------------------------------------------------------
# # Visual Data check (commented out)
View(obesity)
View(obesity_predict)

# Data summary
glimpse(obesity) 

# Plot histogram
options(warn = -1)

# BMI predict
discrete <- ggplot(data = obesity, aes(x = BMI)) +
  geom_histogram(bins=30)
continuous <- ggplot(data = obesity, aes(x = BMI)) +
  stat_density()
grid.arrange(arrangeGrob(discrete, continuous, ncol=2, nrow=1), heights=c(2,1))

# Check summary
summary(obesity)
summary(obesity_predict)

# Prune variable Age as it is not iid
obesity <- obesity[c(which(obesity$Age <= max(obesity_predict$Age))),] # drop Age above 45


# Check NAs
if(all(is.na(obesity) == F)){
  print("No missing values (NAs) detected")
} else if (any(is.na(obesity)) == T) {
  print("Missing values detected")
}

if(all(is.na(obesity_predict) == F)){
  print("No missing values (NAs) detected")
} else if (any(is.na(obesity_predict)) == T) {
  print("Missing values detected")
}


# Transformation of non-numeric variables
variables_non_numeric <- c()
for(col in colnames(obesity)){
  if(!is.numeric(obesity[[col]])==T && any(unique(obesity[[col]])=="Always")==T){
    # Create ordinal scaled variables
    obesity[[col]] <- factor(obesity[[col]],
                             ordered = T,
                             levels = c("no",
                                        "Sometimes",
                                        "Frequently",
                                        "Always"))
    obesity_predict[[col]] <- factor(obesity_predict[[col]],
                                     ordered = T,
                                     levels = c("no",
                                                "Sometimes",
                                                "Frequently",
                                                "Always"))
    variables_non_numeric <- c(variables_non_numeric, col)
  } else if (!is.numeric(obesity[[col]])==T) {
    # Create binary and nominal variables
    obesity[[col]] <- factor(obesity[[col]])
    obesity_predict[[col]] <- factor(obesity_predict[[col]])
    variables_non_numeric <- c(variables_non_numeric, col)
  } else if (is.numeric(obesity[[col]])==T && length(unique(obesity[[col]])) <= 4){
    # Create ordinal scaled variables 
    obesity[[col]] <- factor(obesity[[col]],
                             levels = sort(unique(obesity[[col]]), decreasing = T),
                             ordered = T)
    obesity_predict[[col]] <- factor(obesity_predict[[col]],
                                     levels = sort(unique(obesity_predict[[col]]), decreasing = T),
                                     ordered = T)
  }
}

# Create train and test data
train <- model.matrix(~ ., 
                      data=obesity, 
                      contrast.arg = lapply(sapply(obesity, is.factor),
                                            contrasts, 
                                            contrasts = FALSE))[,-1]

train <- train[,-ncol(train)] # Drop dependent variable BMI

test <- model.matrix(~ ., 
                     data=obesity_predict, 
                     contrast.arg = lapply(sapply(obesity_predict, is.factor),
                                           contrasts, 
                                           contrasts = FALSE))[,-1]

# Correlation analysis
options(warn = -1)
cor <- round(cor(train),2) # Look for multicollinearity
corrplot(cor,tl.cex = 0.7, cl.cex=0.7)
options(warn = 0)


# OLS --------------------------------------------------------------------------
ols <- lm(obesity$BMI~.,as.data.frame(train)) # OLS regression
summary(ols)                                  # summary statistic

# # Plot
options(warn = -1)
par(mfrow= c(2,2))
plot(ols)
dev.off()
options(warn = 0)

# Breusch-Pagan Test for heteroskedasticity
bptest(ols)

# Prediction OLS
ols_pred <- as.data.frame(predict(ols, newdata = as.data.frame(test)))
colnames(ols_pred) <- c("OLS prediction")

# # Write CSV (might differ in a Windows environment)
# write.csv(ols_pred, paste0(getwd(),"//Prediction Results//OLS.csv"), row.names=FALSE)


# Post-Lasso -------------------------------------------------------------------
# Group variables
colnames(train)
v.group <- c(1,2,3,4,5,6,6,7,7,7,8,8,8,9,10,10,11,12,12,12,13,13,14,14,14,15,15,15,15)

# Cross validated group Lasso
glasso.cv <- cv.gglasso(x=train,
                        y=obesity$BMI,
                        group=v.group,
                        loss="ls",
                        pred.loss="L2",
                        nfolds=10)

# Model selection
glasso <- coef(glasso.cv, s = "lambda.min") # variable selection

# Tuning Parameter (lambda with smallest in-sample MSE)
glasso.cv$lambda.min

# Post-Lasso regression
post_lasso <- lm(obesity$BMI ~. ,data=as.data.frame(train[,which(glasso!= 0)-1]))
summary(post_lasso)

# Prediction Post-Lasso
glasso_pred <- as.data.frame(predict(post_lasso, newdata = as.data.frame(test)))
colnames(glasso_pred) <- c("GL prediction")

# # Write CSV (might differ in a Windows environment)
# write.csv(ols_pred, paste0(getwd(),"//Prediction Results//GL.csv"), row.names=FALSE)


# Random forest ----------------------------------------------------------------
# Set up CV settings
forest_cv <- trainControl(method = "repeatedcv", number = 10, repeats = 1)

# Random Forest regression
forest <- train(BMI~.,
                obesity,
                method = "ranger",
                trControl = forest_cv,
                metric = "Rsquared")

# Tuning parameter 
forest$bestTune # mtry: number of variables that are considered as split candidates
                # min.nodes: Minimum number of nodes within a single tree of a Forest

# Prediction Random Forest
forest_pred <- as.data.frame(predict(forest,obesity_predict))
colnames(forest_pred) <- c("RF prediction")

# Write CSV (might differ in a Windows environment)
write.table(forest_pred, paste0(getwd(),"//Prediction Results//lauritz_storch.csv"), 
          row.names = F,
          sep = ",")









# Appendix ---------------------------------------------------------------------
# Check values (true BMI)
row_vector <- rep(NA,length(rownames(obesity_predict)))
for (i in seq(length(rownames(obesity_predict)))){
  row_vector[i] <- which(rownames(obesity_predict)[i] == rownames(obesity))
}
true_BMI <- matrix(obesity$BMI[row_vector])

# Reconstructed BMI in test data
obesity2 <- obesity
obesity_predict2 <- obesity_predict
obesity_predict2$BMI <- true_BMI

# Reconstruct Weight for both data sets
obesity2$Weight <- obesity$BMI*(obesity$Height)^2
obesity_predict2$Weight <- obesity_predict2$BMI*(obesity_predict2$Height)^2

train2 <- model.matrix(~ ., 
                      data=obesity2, 
                      contrast.arg = lapply(sapply(obesity2, is.factor),
                                            contrasts, 
                                            contrasts = FALSE))[,-1]

train2 <- train2[,-(ncol(train2)-1)] # Drop dependent variable BMI

test2 <- model.matrix(~ ., 
                     data=obesity_predict2, 
                     contrast.arg = lapply(sapply(obesity_predict2, is.factor),
                                           contrasts, 
                                           contrasts = FALSE))[,-1]
test2 <- test2[,-(ncol(test2)-1)]

# OLS ------
ols2 <- lm(obesity2$BMI~.,as.data.frame(train2)) # OLS regression
summary(ols2)                                  # summary statistic

# # Plot
par(mfrow= c(2,2))
plot(ols2)
dev.off()

# Post-Lasso -----
k <- 2 # number of variables to select for post-lasso regression
lasso.cv <- cv.glmnet(train2, 
                      y = obesity2$BMI,
                      alpha = 1,
                      standardize = T,
                      family = "gaussian",
                      type.measure = "mse",
                      nfolds = 10,
                      pmax = k)

lasso <- coef(lasso.cv, s = "lambda.min") # variable selection

post_lasso2 <- lm(obesity2$BMI ~. ,data=as.data.frame(train2[,which(lasso!= 0)-1]))
summary(post_lasso2) 

report_regression <- function(model, format, ...){
  if(format == "latex"){
    require(stargazer)
    stargazer(model, ...)
  } else {
    print("This only works with latex output") 
  }
}

report_regression(post_lasso2, format = "latex")
