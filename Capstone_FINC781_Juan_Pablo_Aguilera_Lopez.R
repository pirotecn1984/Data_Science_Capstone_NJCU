#Capstone Data Science FINC-781
##Juan Pablo Aguilera López
###House Price Prediction Using Regression Analysis Techniques


#Data Wrangling and Exploratory Data Analysis
##Packages
library(grid)
library(gridExtra)
library(knitr)
library(mice)
library(ggplot2)
library(reshape2)
library(DMwR2)
library(rpart)
library(rpart.plot)
library(tidyverse)
library(plyr)
library(dplyr)
library(readxl)
library(ISLR)
library(MASS)
library(leaps)
library(caret)
library(corrplot)
library(kernlab)
library(glmnet)
library(pls)
library(randomForest)
library(gbm)
library(xgboost)
library(gridExtra)
library(scales)
library(ggrepel)
library(psych)
library(Hmisc)
library(DataExplorer)
library(stringr)
library(ggpubr)
library(Rmisc)

##Read File
df_train <- read.csv("C:/Users/Juan Pablo/Desktop/NJCU/Fall2019/FINC781_Capstone/House_Prices_Kaggle_Competition/train.csv", header=TRUE, stringsAsFactors = F)
df_train <- data.frame(df_train)
df_test <- read.csv("C:/Users/Juan Pablo/Desktop/NJCU/Fall2019/FINC781_Capstone/House_Prices_Kaggle_Competition/test.csv", header=TRUE, stringsAsFactors = F)
df_test <- data.frame(df_test)

##First Exploratorion of Data and Summary
any(is.na(df_test))
any(is.na(df_train))
glimpse(df_train)
head(df_train)
summary(df_train)

###Drop Id Column
test_labels <- df_test$Id
df_test$Id <- NULL
df_train$Id <- NULL
df_test$SalePrice <- NA
df2 <- rbind(df_train, df_test)
dim(df2)
any(is.na(df2))
glimpse(df2)
summary(df2)

##Exploring Response Variable (Sale Price)
summary(df2$SalePrice)
ggplot(data = df2[!is.na(df2$SalePrice),], aes(x=SalePrice)) +
  geom_histogram(fill = "red", binwidth = 10000) +
  scale_x_continuous(breaks = seq(0, 800000, by=100000), labels = comma)

###Correlation with Sale Price
num_var <- which(sapply(df2, is.numeric)) 
num_names <- names(num_var) 
print(num_names)
df2_num_var <- df2[, num_var]
cor_num_var <- cor(df2_num_var, use="pairwise.complete.obs") 
corr_sorted <- as.matrix(sort(cor_num_var[,'SalePrice']), decreasing = "TRUE")
high_cor <- names(which(apply(corr_sorted, 1, function(x) abs(x) > 0.49))) #Select Correlations above 0.49
cor_num_var <- cor_num_var[high_cor, high_cor]
corrplot.mixed(cor_num_var, tl.col="black", tl.pos = "lt")

#Overall Quality and General Living Area correlation with Sale Price
###Overall Quality
ggplot(data = df2[!is.na(df2$SalePrice),], aes(x = factor(OverallQual), y = SalePrice))+
  geom_boxplot(col='red') + labs(x = 'Overall Quality') +
  scale_y_continuous(breaks = seq(0, 800000, by = 100000), labels = comma)

###General Living Area
ggplot(data = df2[!is.na(df2$SalePrice),], aes(x = GrLivArea, y = SalePrice))+
  geom_point(col='red') + geom_smooth(method = "lm", se = FALSE, color = "black", aes(group = 1)) +
  scale_y_continuous(breaks = seq(0, 800000, by = 100000), labels = comma) +
  geom_text_repel(aes(label = ifelse(df2$GrLivArea[!is.na(df2$SalePrice)] > 4400, rownames(df2), '')))

df2[c(1183, 524, 1299), c('SalePrice', 'GrLivArea', 'OverallQual')]   #Outliers exploration

#Linear Model (With Raw Data - Just to get a glimpse)
##Sale Price on Overall Quality
set.seed(3333)
model <- lm(formula = df2$SalePrice ~ df2$OverallQual, na.action = na.omit) 
model
summary(model)
plot(df2$OverallQual, df2$SalePrice, pch = 16, cex = 0.9, col = "red", main = "Sale Price on Overall Quality", xlab = "Overall Quality", ylab = "Sale Price")
abline(lm(df2$SalePrice ~ df2$OverallQual))

##Sale Price on General Living Area
set.seed(3333)
model2 <- lm(formula = df2$SalePrice ~ df2$GrLivArea, na.action = na.omit) 
model2
summary(model2)
plot(df2$GrLivArea, df2$SalePrice, pch = 16, cex = 0.9, col = "blue", main = "Sale Price on General Living Area", xlab = "General Living Area", ylab = "Sale Price")
abline(lm(df2$SalePrice ~ df2$GrLivArea))

##Data Cleaning, Missing Values, Encoding/Factorizing
missing_values <- which(colSums(is.na(df2)) > 0)  ###Variables where missing data is present
print(missing_values)

###Create a series of vectors for later use in reevaluation of features. E.g. When Quality = "NA", "Poor", "Fair", etc...
quality <- c("None" = 0, "Po" = 1, "Fa" = 2, "TA" = 3, 'Gd' = 4, 'Ex' = 5)
bsm_expo <- c('None'=0, 'No'=1, 'Mn'=2, 'Av'=3, 'Gd'=4)
finish_type <- c('None'=0, 'Unf'=1, 'LwQ'=2, 'Rec'=3, 'BLQ'=4, 'ALQ'=5, 'GLQ'=6)
functionality <- c('Sal'=0, 'Sev'=1, 'Maj2'=2, 'Maj1'=3, 'Mod'=4, 'Min2'=5, 'Min1'=6, 'Typ'=7)
gar_finish <- c('None'=0, 'Unf'=1, 'RFn'=2, 'Fin'=3)
st_vec <- c('Grvl'=0, 'Pave'=1)
pv_dr <- c('N'=0, 'P'=1, 'Y'=2)
land_sl <- c('Sev'=0, 'Mod'=1, 'Gtl'=2)
lt_shape <- c('IR1' = 0, 'IR2'=1, 'IR3'=3, 'Reg' = 4)
lt_con <- c('FR3'=0, 'FR2'=1,'Corner'=3, 'Inside'=4, 'CulDSac'=5)

###MSZoning 
glimpse(df2$MSZoning)
df2$MSZoning[is.na(df2$MSZoning)] <- names(sort(-table(df2$MSZoning)))[1]
df2$MSZoning <- as.factor(df2$MSZoning)
any(is.na(df2$MSZoning))

###Lot Frontage 
glimpse(df2$LotFrontage)
for (i in 1:nrow(df2)){
  if(is.na(df2$LotFrontage[i])){
    df2$LotFrontage[i] <- as.integer(median(df2$LotFrontage[df2$Neighborhood == df2$Neighborhood[i]], na.rm=TRUE)) 
  }
}
any(is.na(df2$LotFrontage))

###Alley
glimpse(df2$Alley)
df2$Alley[is.na(df2$Alley)] <- 'None'
df2$Alley <- as.factor(df2$Alley)
any(is.na(df2$Alley))

###Utilities    
glimpse(df2$Utilities)
df2$Utilities[is.na(df2$Utilities)] <- 'None'      
df2$Utilities <- as.factor(df2$Utilities)          
any(is.na(df2$Utilities))
df2$Utilities <- NULL   ###All houses except for one have 'AllPub' - Get rid of variable since it is useless for prediction.


### Exterior1st
glimpse(df2$Exterior1st)
df2$Exterior1st[is.na(df2$Exterior1st)] <- mode((-table(df2$Exterior1st)))[1]
df2$Exterior1st <- as.factor(df2$Exterior1st)
any(is.na(df2$Exterior1st))

###Exterior2nd
glimpse(df2$Exterior2nd)
df2$Exterior2nd[is.na(df2$Exterior2nd)] <- mode((-table(df2$Exterior2nd)))[1]
df2$Exterior2nd <- as.factor(df2$Exterior2nd)
any(is.na(df2$Exterior2nd))

###MasVnrType & MasVnrArea (Masonry)
glimpse(df2$MasVnrType)
glimpse(df2$MasVnrArea)
df2[is.na(df2$MasVnrType) & !is.na(df2$MasVnrArea), c('MasVnrType', 'MasVnrArea')] ###Find single house with NA in only MasVnrType
df2$MasVnrType[2611] <- names(-table(df2$MasVnrType))[2]  ###BrkFace value for that particular house
df2$MasVnrType[is.na(df2$MasVnrType)] <- "None"  ###Input 'None' for NAs
mas_var <- c("None" = 0, "BrkCmn" = 1, "BrkFace" = 2, "Stone" = 3)  ###Create mumbers for different types of masonry
df2$MasVnrType <- as.integer(revalue(df2$MasVnrType, mas_var))   ###Assign numbers for different types of masonry
df2$MasVnrArea[is.na(df2$MasVnrArea)] <-0   ###0 for area variable
any(is.na(df2$MasVnrType))
any(is.na(df2$MasVnrArea))

###BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1, BsmtFinSF1, BsmtFinType2, BsmtFinSF2, BsmtUnfSF, TotalBsmtSF, TotalBsmtSF, BsmtFullBath, BsmtHalfBath (Basement)
df2[!is.na(df2$BsmtFinType1) & (is.na(df2$BsmtCond)| is.na(df2$BsmtQual)|is.na(df2$BsmtExposure)|is.na(df2$BsmtFinType2)|is.na(df2$BsmtFinSF1)|is.na(df2$BsmtFinType1)
                                |is.na(df2$BsmtFinSF2)|is.na(df2$BsmtUnfSF)|is.na(df2$BsmtFullBath)
                                |is.na(df2$BsmtHalfBath)), c('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'BsmtFinSF1',
                                                             'BsmtFinType1', 'BsmtFinSF2', 'BsmtUnfSF', 'BsmtFullBath', 'BsmtHalfBath')]
###Assign Values to NAs in variables and convert to vectors
df2$BsmtFinType2[333] <- names(sort(-table(df2$BsmtFinType2)))[1]   
df2$BsmtFinType2[is.na(df2$BsmtFinType2)] <- 'None'
df2$BsmtFinType2 <- as.integer(revalue(df2$BsmtFinType2, finish_type))
any(is.na(df2$BsmtFinType2))

df2$BsmtExposure[c(949, 1488, 2349)] <- names(sort(-table(df2$BsmtExposure)))[1]
df2$BsmtExposure[is.na(df2$BsmtExposure)] <- 'None'
df2$BsmtExposure <- as.integer(revalue(df2$BsmtExposure, bsm_expo))
any(is.na(df2$BsmtExposure))

df2$BsmtCond[c(2041, 2186, 2525)] <- names(sort(-table(df2$BsmtCond)))[1]
df2$BsmtCond[is.na(df2$BsmtCond)] <- 'None'
df2$BsmtCond <- as.integer(revalue(df2$BsmtCond, quality))
any(is.na(df2$BsmtCond))

df2$BsmtQual[c(2218, 2219)] <- names(sort(-table(df2$BsmtQual)))[1]
df2$BsmtQual[is.na(df2$BsmtQual)] <- 'None'
df2$BsmtQual <- as.integer(revalue(df2$BsmtQual, quality))
any(is.na(df2$BsmtQual))

df2$BsmtFinType1[is.na(df2$BsmtFinType1)] <- 'None'
df2$BsmtFinType1 <- as.integer(revalue(df2$BsmtFinType1, finish_type))
any(is.na(df2$BsmtFinType1))

df2$BsmtHalfBath[is.na(df2$BsmtHalfBath)] <-0
any(is.na(df2$BsmtHalfBath))

df2$BsmtFullBath[is.na(df2$BsmtFullBath)] <-0
any(is.na(df2$BsmtFullBath))

df2$BsmtUnfSF[is.na(df2$BsmtUnfSF)] <-0
any(is.na(df2$BsmtUnfSF))
 
df2$TotalBsmtSF[is.na(df2$TotalBsmtSF)] <- 0
any(is.na(df2$TotalBsmtSF))

df2$BsmtFinSF1[is.na(df2$BsmtFinSF1)] <- 0
any(is.na(df2$BsmtFinSF1))

df2$BsmtFinSF2[is.na(df2$BsmtFinSF2)] <- 0
any(is.na(df2$BsmtFinSF2))

###Electrical
df2$Electrical[is.na(df2$Electrical)] <- names(sort(-table(df2$Electrical)))[1]
df2$Electrical <- as.factor(df2$Electrical)
any(is.na(df2$Electrical))

###KitchenQual
glimpse(df2$KitchenQual)
df2$KitchenQual[is.na(df2$KitchenQual)] <- 'TA' 
df2$KitchenQual<-as.integer(revalue(df2$KitchenQual, quality))
any(is.na(df2$KitchenQual))

###Functional
glimpse(df2$Functional)
df2$Functional[is.na(df2$Functional)] <- names(sort(-table(df2$Functional)))[1]
df2$Functional <- as.integer(revalue(df2$Functional, functionality))
any(is.na(df2$Functional))

###FireplaceQu
glimpse(df2$FireplaceQu)
df2$FireplaceQu[is.na(df2$FireplaceQu)] <- 'None'
df2$FireplaceQu<-as.integer(revalue(df2$FireplaceQu, quality))
any(is.na(df2$FireplaceQu))

###GarageType, GarageYrBlt, GarageFinish, GarageCars, GarageArea, GarageQual, GarageCond

df2$GarageYrBlt[is.na(df2$GarageYrBlt)] <- df2$YearBuilt[is.na(df2$GarageYrBlt)]  ###Replace Garage YR Built NAs by the year the unit was built

length(which(is.na(df2$GarageType) & is.na(df2$GarageFinish) & is.na(df2$GarageCond) & is.na(df2$GarageQual)))

kable(df2[!is.na(df2$GarageType) & is.na(df2$GarageFinish),
          c('GarageCars', 'GarageArea', 'GarageType', 'GarageCond', 'GarageQual', 'GarageFinish')])  ###Find additional NAs

df2$GarageCars[2577] <- 0
df2$GarageArea[2577] <- 0
df2$GarageType[2577] <- NA

df2$GarageQual[2127] <- names(sort(-table(df2$GarageQual)))[1]
df2$GarageCond[2127] <- names(sort(-table(df2$GarageCond)))[1]
df2$GarageFinish[2127] <- names(sort(-table(df2$GarageFinish)))[1]

df2$GarageFinish[is.na(df2$GarageFinish)] <- 'None'
df2$GarageFinish<-as.integer(revalue(df2$GarageFinish, gar_finish))
any(is.na(df2$GarageFinish))

df2$GarageType[is.na(df2$GarageType)] <- 'No Garage'
df2$GarageType <- as.factor(df2$GarageType)
any(is.na(df2$GarageType))

df2$GarageCond[is.na(df2$GarageCond)] <- 'None'
df2$GarageCond<-as.integer(revalue(df2$GarageCond, quality))
any(is.na(df2$GarageCond))

df2$GarageQual[is.na(df2$GarageQual)] <- 'None'
df2$GarageQual<-as.integer(revalue(df2$GarageQual, quality))
any(is.na(df2$GarageQual))

###PoolQC
glimpse(df2$PoolQC)
df2$PoolQC[is.na(df2$PoolQC)] <- 'None'
df2$PoolQC<-as.integer(revalue(df2$PoolQC, quality))
any(is.na(df2$PoolQC))

###Fence
glimpse(df2$Fence)
df2$Fence[is.na(df2$Fence)] <- 'None'
df2$Fence <- as.factor(df2$Fence)
any(is.na(df2$Fence))

###MiscFeature
glimpse(df2$MiscFeature)
df2$MiscFeature[is.na(df2$MiscFeature)] <- 'None'
df2$MiscFeature <- as.factor(df2$MiscFeature)
any(is.na(df2$MiscFeature))

###SaleType
glimpse(df2$SaleType)
df2$SaleType[is.na(df2$SaleType)] <- names(sort(-table(df2$SaleType)))[1]
df2$SaleType <- as.factor(df2$SaleType)
any(is.na(df2$SaleType))

###SaleCondition
glimpse(df2$SaleCondition)
df2$SaleCondition <- as.factor(df2$SaleCondition)
any(is.na(df2$SaleCondition))

###GarageYrBlt
glimpse(df2$GarageYrBlt)
df2$GarageYrBlt[is.na(df2$GarageYrBlt)] <- df2$YearBuilt[is.na(df2$GarageYrBlt)]  ###Replace Garage Year Built observations with House Year Built
any(is.na(df2$GarageYrBlt))

missing_values <- which(colSums(is.na(df2)) > 0)  ###Check again for any missing values 
print(missing_values)

##Character Variables
chrt_feaures <- names(df2[,sapply(df2, is.character)])
print(chrt_feaures)

###Street
glimpse(df2$Street)
df2$Street<-as.integer(revalue(df2$Street, st_vec))

###LotShape
glimpse(df2$LotShape)
df2$LotShape <- as.factor(revalue(df2$LotShape, lt_shape))

###LandContour
glimpse(df2$LandContour)
df2$LandContour <- as.factor(df2$LandContour)

###LotConfig
glimpse(df2$LotConfig)
df2$LotConfig <- as.factor(revalue(df2$LotConfig, lt_con))

###LandSlope
glimpse(df2$LandSlope)
df2$LandSlope<-as.integer(revalue(df2$LandSlope, land_sl))

###Neighborhood
glimpse(df2$Neighborhood)
df2$Neighborhood <- as.factor(df2$Neighborhood)

###Condition1
glimpse(df2$Condition1)
df2$Condition1 <- as.factor(df2$Condition1)

###Condition2
glimpse(df2$Condition2)
df2$Condition2 <- as.factor(df2$Condition2)

###BldgType
glimpse(df2$BldgType)
df2$BldgType <- as.factor(df2$BldgType)

###HouseStyle
glimpse(df2$HouseStyle)
df2$HouseStyle <- as.factor(df2$HouseStyle)

###RoofStyle
glimpse(df2$RoofStyle)
df2$RoofStyle <- as.factor(df2$RoofStyle)

###RoofMatl
glimpse(df2$RoofMatl)
df2$RoofMatl <- as.factor(df2$RoofMatl)

###ExterQual
glimpse(df2$ExterQual)
df2$ExterQual <- as.integer(revalue(df2$ExterQual, quality))

###ExterCond
glimpse(df2$ExterCond)
df2$ExterCond <- as.integer(revalue(df2$ExterCond, quality))

###Foundation
glimpse(df2$Foundation)
df2$Foundation <- as.factor(df2$Foundation)

###Heating
glimpse(df2$Heating)
df2$Heating <- as.factor(df2$Heating)

###HeatingQC
glimpse(df2$HeatingQC)
df2$HeatingQC <- as.integer(revalue(df2$HeatingQC, quality))

###CentralAir
glimpse(df2$CentralAir)
df2$CentralAir <- as.integer(ifelse(df2$CentralAir=="Y", 1, 0))

###PavedDrive
glimpse(df2$PavedDrive)
df2$PavedDrive<-as.integer(revalue(df2$PavedDrive, pv_dr))

##Check for character Variables again
chrt_feaures <- names(df2[,sapply(df2, is.character)])
print(chrt_feaures)

##Converting other variables into factors

###Month and Year Sold
df2$MoSold <- as.factor(df2$MoSold)
glimpse(df2$MoSold)
glimpse(df2$YrSold)

###Visualization of Sales Prices with Year and Month Sold
year_sold <- ggplot(df2[!is.na(df2$SalePrice),], aes(x = as.factor(YrSold), y = SalePrice)) +
  geom_bar(stat = 'summary', fun.y = "median", fill = 'red')+
  scale_y_continuous(breaks = seq(0, 800000, by = 25000), labels = comma) +
  geom_label(stat = "count", aes(label = ..count.., y = ..count..)) +
  coord_cartesian(ylim = c(0, 200000)) +
  geom_hline(yintercept = 163000, linetype = "dashed", color = "black") 

month_sold <- ggplot(df2[!is.na(df2$SalePrice),], aes(x = MoSold, y = SalePrice)) +
  geom_bar(stat = 'summary', fun.y = "median", fill='red')+
  scale_y_continuous(breaks= seq(0, 800000, by = 25000), labels = comma) +
  geom_label(stat = "count", aes(label = ..count.., y = ..count..)) +
  coord_cartesian(ylim = c(0, 200000)) +
  geom_hline(yintercept = 163000, linetype = "dashed", color = "black") 

grid.arrange(year_sold, month_sold, widths = c(1,2))

###MSSubClass
glimpse(df2$MSSubClass)
df2$MSSubClass <- as.factor(df2$MSSubClass)
df2$MSSubClass<-revalue(df2$MSSubClass, c('20'='1-STORY 1946 & NEWER ALL STYLES', '30'='1-STORY 1945 & OLDER',
                                          '40'='1-STORY W/FINISHED ATTIC ALL AGES', '45'='1-1/2 STORY - UNFINISHED ALL AGES',
                                          '50'='1-1/2 STORY FINISHED ALL AGES', '60'='2-STORY 1946 & NEWER', '70'='2-STORY 1945 & OLDER',
                                          '75'='2-1/2 STORY ALL AGES', '80'='SPLIT OR MULTI-LEVEL', '85'='SPLIT FOYER',
                                          '90'='DUPLEX - ALL STYLES AND AGES','120'='1-STORY PUD 1946 & NEWER',
                                          '150'='1-1/2 STORY PUD - ALL AGES', '160'='2-STORY PUD - 1946 & NEWER',
                                          '180'='PUD - MULTILEVEL - INCL SPLIT LEV/FOYER', '190'='2 FAMILY CONVERSION - ALL STYLES AND AGES'))
glimpse(df2$MSSubClass)

dim(df2)
summary(df2)

##Correlation Once the data set is cleaned 
num_var <- which(sapply(df2, is.numeric)) 
num_names <- names(num_var) 
factor_var <- which(sapply(df2, is.factor))
print(num_names)
print(factor_var)
df2_num_var <- df2[, num_var]
cor_num_var <- cor(df2_num_var, use="pairwise.complete.obs") 
corr_sorted <- as.matrix(sort(cor_num_var[,'SalePrice']), decreasing = "TRUE")
high_cor <- names(which(apply(corr_sorted, 1, function(x) abs(x) > 0.49))) #Select Correlations above 0.49
cor_num_var <- cor_num_var[high_cor, high_cor]
corrplot.mixed(cor_num_var, tl.col="black", tl.pos = "lt", tl.cex = 0.7,cl.cex = .7, number.cex=.7)


#Random Forest for variable importance
set.seed(1111)
houses_rf <- randomForest(x = df2[1:1460,-79], y = df2$SalePrice[1:1460], ntree = 100, importance=TRUE)
importance_rf <- importance(houses_rf)
importance_rf <- data.frame(variables = row.names(importance_rf), MSE = importance_rf[,1])
importance_rf <- importance_rf[order(importance_rf$MSE, decreasing = TRUE),]

ggplot(importance_rf[1:10,], aes(x = reorder(variables, MSE), y = MSE, fill = MSE)) + geom_bar(stat = 'identity') + 
  labs(x = 'Features', y= 'MSE Increase')


##Feature Engineering
###Binding Porch Variables
df2$Porchs_sq <- df2$OpenPorchSF + df2$EnclosedPorch + df2$X3SsnPorch + df2$ScreenPorch + df2$WoodDeckSF
glimpse(df2$Porchs_sq)

cor(df2$SalePrice, df2$Porchs_sq, use= "pairwise.complete.obs")

###Correlation of Total Porch Sqft with Sale Price
ggplot(data = df2[!is.na(df2$SalePrice),], aes(x = Porchs_sq, y = SalePrice)) +
  geom_point(col = 'red') + geom_smooth(method = "lm", se = FALSE, color = "black", aes(group = 1)) +
  scale_y_continuous(breaks = seq(0, 800000, by = 100000), labels = comma)

##Adding variables (House Age, New House, and Remodeled House)
glimpse(df2$YrSold)
glimpse(df2$YearRemodAdd)
df2$HouseAge <- as.numeric(df2$YrSold) - df2$YearRemodAdd

cor(df2$SalePrice[!is.na(df2$SalePrice)], df2$HouseAge[!is.na(df2$SalePrice)])  ###Correlation between House Age and Price

ggplot(data = df2[!is.na(df2$SalePrice),], aes(x = HouseAge, y = SalePrice))+
  geom_point(col = 'red') + geom_smooth(method = "lm", se = FALSE, color = "black", aes(group = 1)) +
  scale_y_continuous(breaks = seq(0, 800000, by = 100000), labels = comma)


df2$NewHouse <- ifelse(df2$YrSold == df2$YearBuilt, 1, 0) #1 = New Constraction, 0 = Old House
summary(df2$SalePrice)

ggplot(df2[!is.na(df2$SalePrice),], aes(x = as.factor(NewHouse), y = SalePrice)) +
  geom_bar(stat ='summary', fun.y = "mean", fill = "red") +
  geom_label(stat = "count", aes(label = ..count.., y = ..count..), size = 8) +
  scale_y_continuous(breaks= seq(0, 800000, by = 50000), labels = comma) +
  theme_grey(base_size = 18) +
  geom_hline(yintercept = 180921, linetype = "solid")

df2$HouseRemodel <- ifelse(df2$YearBuilt == df2$YearRemodAdd, 0, 1) #0 = No Remodeled, 1 = Remodeled

ggplot(df2[!is.na(df2$SalePrice),], aes(x = as.factor(HouseRemodel), y = SalePrice)) +
  geom_bar(stat = 'summary', fun.y = "mean", fill = 'red') +
  geom_label(stat = "count", aes(label = ..count.., y = ..count..), size = 8) +
  scale_y_continuous(breaks = seq(0, 800000, by = 50000), labels = comma) +
  theme_grey(base_size = 16) +
  geom_hline(yintercept = 180921, linetype = "solid")

df2$YrSold <- as.factor(df2$YrSold)  

##Binding Bathroom variables
df2$TotalBathrooms <-  (df2$HalfBath * 0.5) + (df2$BsmtHalfBath * 0.5) + df2$FullBath + df2$BsmtFullBath  ###Count half bath as 0.5

ggplot(data = df2, aes(x = as.factor(TotalBathrooms))) +
  geom_histogram(stat = 'count')

cor(df2$SalePrice[!is.na(df2$SalePrice)], df2$TotalBathrooms[!is.na(df2$SalePrice)])  ###Correlation between total number of bathrooms and Sale Price

ggplot(data = df2[!is.na(df2$SalePrice),], aes(x = TotalBathrooms, y = SalePrice))+
  geom_point(col = 'red') + geom_smooth(method = "lm", se = FALSE, color = "black", aes(group = 1)) +
  scale_y_continuous(breaks = seq(0, 800000, by = 100000), labels = comma)

##Adding Square Fotage into a new variable (Total Sqft of living space)
df2$TotalSQFT <- df2$GrLivArea + df2$TotalBsmtSF

cor(df2$SalePrice[!is.na(df2$SalePrice)], df2$TotalSQFT[!is.na(df2$SalePrice)])

ggplot(data = df2[!is.na(df2$SalePrice),], aes(x = TotalSQFT, y = SalePrice))+
  geom_point(col = 'red') + geom_smooth(method = "lm", se = FALSE, color = "black", aes(group = 1)) +
  scale_y_continuous(breaks = seq(0, 800000, by = 100000), labels = comma)

###Correaltion without the two potential outliers
cor(df2$SalePrice[-c(1299, 524)], df2$TotalSQFT[-c(1299, 524)], use= "pairwise.complete.obs")

##Neighborhood Variables (Binning)
summary(df2$SalePrice)

ggplot(df2[!is.na(df2$SalePrice),], aes(x = reorder(Neighborhood, SalePrice, FUN = mean), y = SalePrice)) +
  geom_bar(stat = 'summary', fun.y = "mean", fill = 'red') + labs(x = 'Neighborhood Name', y = "Mean Sale Price") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  scale_y_continuous(breaks = seq(0, 800000, by = 50000), labels = comma) +
  geom_label(stat = "count", aes(label = ..count.., y = ..count..), size = 4) +
  geom_hline(yintercept = 180921, linetype = "solid", color = "black")

table(df2$Neighborhood)

df2$Nhood_Afft[df2$Neighborhood %in% c('StoneBr', 'NridgHt', 'NoRidge')] <- 5
df2$Nhood_Afft[df2$Neighborhood %in% c('Crawfor', 'ClearCr', 'Somerst', 'Veenker', 'Timber')] <- 4
df2$Nhood_Afft[df2$Neighborhood %in% c('SawyerW', 'NWAmes', 'Gilbert', 'Blmngtn', 'CollgCr')] <- 3
df2$Nhood_Afft[df2$Neighborhood %in% c('BrkSide', 'Sawyer', 'SWISU', 'NPkVill', 'NAmes', 'Mitchel', 'Edwards', 'OldTown', 'Blueste')] <- 2
df2$Nhood_Afft[df2$Neighborhood %in% c('MeadowV', 'IDOTRR', 'BrDale')] <- 1


table(df2$Nhood_Afft)
summary(df2$Nhood_Afft)
any(is.na(df2$Nhood_Afft))

##Correlation again
num_var <- which(sapply(df2, is.numeric)) 
num_names <- names(num_var) 
factor_var <- which(sapply(df2, is.factor))
print(num_names)
print(factor_var)
df2_num_var <- df2[, num_var]
cor_num_var <- cor(df2_num_var, use="pairwise.complete.obs") 
corr_sorted <- as.matrix(sort(cor_num_var[,'SalePrice']), decreasing = "TRUE")
high_cor <- names(which(apply(corr_sorted, 1, function(x) abs(x) > 0.49))) #Select Correlations above 0.49
cor_num_var <- cor_num_var[high_cor, high_cor]
corrplot.mixed(cor_num_var, tl.col="black", tl.pos = "lt", tl.cex = 0.7,cl.cex = .7, number.cex=.7)


#Drop Highly Correlated variables
dim(df2)
df2 <- subset(df2, select=-c(OpenPorchSF, EnclosedPorch, X3SsnPorch, ScreenPorch, WoodDeckSF, YearRemodAdd,
                             HalfBath, BsmtHalfBath, FullBath, BsmtFullBath, X1stFlrSF,TotalBsmtSF,
                             GarageYrBlt, GarageArea, GarageCond, TotRmsAbvGrd, BsmtFinSF1, Neighborhood))  ##Drop Variables highly correlated amongst them
dim(df2)

##Remove Outliers
df2 <- df2[-c(524, 1299),]

##Correlation again
num_var <- which(sapply(df2, is.numeric)) 
num_names <- names(num_var) 
factor_var <- which(sapply(df2, is.factor))
print(num_names)
print(factor_var)
df2_num_var <- df2[, num_var]
cor_num_var <- cor(df2_num_var, use="pairwise.complete.obs") 
corr_sorted <- as.matrix(sort(cor_num_var[,'SalePrice']), decreasing = "TRUE")
high_cor <- names(which(apply(corr_sorted, 1, function(x) abs(x) > 0.49))) #Select Correlations above 0.49
cor_num_var <- cor_num_var[high_cor, high_cor]
corrplot.mixed(cor_num_var, tl.col="black", tl.pos = "lt", tl.cex = 0.7,cl.cex = .7, number.cex=.7)


##Normalizing Numeric Predictors   
df2_num_var <- df2_num_var[!(df2_num_var %in% c('MSSubClass', 'MoSold', 'YrSold', 'SalePrice', 'OverallQual', 'OverallCond'))]
df2_num_var <- append(df2_num_var, c('HouseAge', 'Porchs_sq', 'TotalBathrooms', 'TotalSQFT'))

num_var2 <- df2[, names(df2) %in% df2_num_var]

df2_factors <- df2[, !(names(df2) %in% df2_num_var)]
df2_factors <- df2_factors[, names(df2_factors) != 'SalePrice']

for(i in 1:ncol(num_var2)){
  if (abs(skew(num_var2[,i])) > 0.75){
    num_var2[,i] <- log(num_var2[,i] + 1)
  }
}

predic_num <- preProcess(num_var2, method = c("center", "scale"))

predic_num

df2_normalized <- predict(predic_num, num_var2)

dim(df2_normalized)

##Hot-encoding categorical variables
df2_dummyvars <- as.data.frame(model.matrix(~.-1, df2_factors))

dim(df2_dummyvars)


#check for absent values in the test data
absentcoltest <- which(colSums(df2_dummyvars[(nrow(df2[!is.na(df2$SalePrice),]) + 1):nrow(df2),]) == 0)
colnames(df2_dummyvars[absentcoltest])
df2_dummyvars <- df2_dummyvars[, -absentcoltest] #Removing predictors from test data

#check for absent values in train data
absentcoltrain <- which(colSums(df2_dummyvars[1:nrow(df2[!is.na(df2$SalePrice),]),]) == 0)
colnames(df2_dummyvars[absentcoltrain])
df2_dummyvars <- df2_dummyvars[, -absentcoltrain] #removing predictors from train data

###Removing variables with less than 15 "ones"
little_var <- which(colSums(df2_dummyvars[1:nrow(df2[!is.na(df2$SalePrice),]),]) < 15)
colnames(df2_dummyvars[little_var])

df2_dummyvars <- df2_dummyvars[, -little_var] #removing predictors with less than 15 "1"
dim(df2_dummyvars)

#Renponse variable skwedness
skew(df2$SalePrice)

##Plot
qqnorm(df2$SalePrice)
qqline(df2$SalePrice)

df2$SalePrice <- log(df2$SalePrice + 1) 
skew(df2$SalePrice)

qqnorm(df2$SalePrice)
qqline(df2$SalePrice)




num_predictors <- cbind(df2_normalized, df2_dummyvars) #combining all numeric predictors into a single dataframe
dim(num_predictors)

#Train and test datasets
train1 <- num_predictors[!is.na(df2$SalePrice),]
test1 <- num_predictors[is.na(df2$SalePrice),]

##Random Forest Again
set.seed(1111)
dim(df2)
length(df2$SaleType)
summary(df2)
houses_rf <- randomForest(x = df2[1:1458,-63], y = df2$SalePrice[1:1458], ntree = 100, importance=TRUE, na.action = na.roughfix)
importance_rf <- importance(houses_rf)
importance_rf <- data.frame(variables = row.names(importance_rf), MSE = importance_rf[,1])
importance_rf <- importance_rf[order(importance_rf$MSE, decreasing = TRUE),]

ggplot(importance_rf[1:15,], aes(x = reorder(variables, MSE), y = MSE, fill = MSE)) + geom_bar(stat = 'identity') + 
  labs(x = 'Features', y= 'MSE Increase')

#Linear Model (Sale Price on TotalSQFT)
set.seed(3333)
model <- lm(formula = !is.na(df2$SalePrice) ~ df2$TotalSQFT) 
model
summary(model)
plot(df2$OverallQual, df2$SalePrice, pch = 16, cex = 1.3, col = "blue", main = "SalePrice vs Total SQ Ft", xlab = "Total Sq Ft", ylab = "SalePrice")
abline(lm(df2$SalePrice ~ df2$OverallQual))


#Linear Model (Sale Price on OverallQual)
set.seed(3333)
model1 <- lm(formula = !is.na(df2$SalePrice) ~ df2$OverallQual) 
model1
summary(model1)
plot(df2$OverallQual, df2$SalePrice, pch = 16, cex = 1.3, col = "blue", main = "SalePrice vs Overall Quality", xlab = "Overall Quality", ylab = "SalePrice")
abline(lm(df2$SalePrice ~ df2$OverallQual))

#Multiple Regression
model3 <- lm(formula = !is.na(SalePrice) ~ TotalSQFT + OverallQual + GrLivArea + Nhood_Afft, data=df2)
model3
summary(model3)






#Lasso
set.seed(4444)
control_var <- trainControl(method = "cv", number = 10)
lassoGrid <- expand.grid(alpha = 1, lambda = seq(0.001,0.1,by = 0.0005))

#Lasso
lasso_model <- train(x=train1, y=df2$SalePrice[!is.na(df2$SalePrice)], method = 'glmnet', 
                     trControl= control_var, tuneGrid=lassoGrid) 
lasso_model$bestTune
#Number of selected variables in lasso model
lasso_variables <- varImp(lasso_model, scale = F)
lasso_importance <- lasso_variables$importance

vars_selection <- length(which(lasso_importance$Overall!= 0))
vars_notused <- length(which(lasso_importance$Overall == 0))

print(vars_selection)  #Number of variables used
print(vars_notused)

#Print model
print(lasso_model)
print(lasso_model$bestTune)
min(lasso_model$results$RMSE)

print(paste0('Lasso best parameters: ' , lasso_model$finalModel$lambdaOpt))

#Predictions
lasso_predictions <- predict(lasso_model, test1)
lasso2 <- exp(lasso_predictions) #Convert the log back to real values
head(lasso2, 10)
tail(lasso2,10)
summary(lasso2)

#Plot the fractions to see how the fraction was chosen (minimum RMSE)
plot(lasso_model)
plot(lasso2)

x<-predict.elnet(lasso_model$finalModel, type='coefficients', 
             s=lasso_model$bestTune$fraction, mode='fraction')
head(sort(x, decreasing = F))
#Ridge regression using caret package
#Train the model on the training dataset and automatically select lambda 
set.seed(3333)
ridge_fit <- train(x=train1, y=df2$SalePrice[!is.na(df2$SalePrice)], method = 'ridge', 
      trControl= control_var) 

#Print model
print(ridge_fit)
print(ridge_fit$bestTune)
min(ridge_fit$results$RMSE)
print(paste0('Ridge best parameters: ' , ridge_fit$finalModel$lambdaOpt))

#Predictions
ridge_predictions <- predict(ridge_fit, test1)
ridge2 <- exp(ridge_predictions) #Convert the log back to real values
head(ridge2, 10)
tail(ridge2,10)
summary(ridge2)

#Plot the lambdas to see how the lambda was chosen (minimum RMSE)
plot(ridge_fit)



##Linear Model
linear<-train(y = df2$SalePrice[!is.na(df2$SalePrice)], 
              x = train1, 
              method = 'lm',
              metric =  "Rsquared"
)
summary(linear)
print(linear)
plot(linear)





#Bagging
#No tuning parameters supported
bag_fit <- train(x=train1, y=df2$SalePrice[!is.na(df2$SalePrice)], data = train1, method = "treebag",
                 trControl=control_var)
bag_fit
predictions <- predict(bag_fit, newdata = test1)

#To see the importance of the variables
bagImp <- varImp(bag_fit, scale=TRUE)
bagImp
plot(bagImp)
plot(bag_fit)

##Fitting Classification Trees
tree_fit <- train(x = train1, y = df2$SalePrice[!is.na(df2$SalePrice)], 
               method = "rpart", 
               trControl = control_var, na.action=na.omit)
#To see the tuned complexity parameter (Gini Coeff)
tree_fit$bestTune
#To see the tree splits
tree_fit$finalModel
#Plot complexity parameter tuning runs
plot(tree_fit)
#Plot the tree
library(rattle)
library(rpart.plot)
library(RColorBrewer)
fancyRpartPlot(tree_fit$finalModel)
#Predict
predictions <- predict(tree_fit, newdata = test1)
predictions2 <- exp(predictions)
fancyRpartPlot(predictions2)
#To see the importance of the variables
treeImp <- varImp(tree_fit, scale = TRUE)
treeImp
plot(treeImp)

# Support Vector Machine
set.seed(3333)
#Fit the SVM model to training data. Radial Kernel.  
#Cost parameter C = default
#Gamma (sigma) = .2
modSVMFit <- train(x = train1, y = df2$SalePrice[!is.na(df2$SalePrice)], 
                  method = "svmRadial", sigma = .2, 
                  trControl = control_var)

plot(modSVMFit)
#See model fit details
modSVMFit$finalModel
#See the tuning parametrs used 
modSVMFit$bestTune
#See the results details by each optimization run
modSVMFit$results
#Predict test dataset
SVMpredict <- predict(modSVMFit,test1)
SVMpredict2 <- exp(SVMpredict)


#AVERAGING PREDICTIONS!!!!!!!!!FIX!!!!!!!!!!!
sub_avg1 <- data.frame(Id = test_labels, SalePrice = lasso2)
sub_avg2 <- data.frame(Id = test_labels, SalePrice = ridge2)
sub_avg3 <- data.frame(Id = test_labels, SalePrice = predictions2)
sub_avg4 <- data.frame(Id = test_labels, SalePrice = SVMpredict2)
head(sub_avg, 10)
tail(sub_avg, 10)
write.csv(sub_avg1, file = 'linear.csv', row.names = F)
write.csv(sub_avg1, file = 'lasso.csv', row.names = F)
write.csv(sub_avg2, file = 'ridge.csv', row.names = F)
write.csv(sub_avg3, file = 'tree.csv', row.names = F)
write.csv(sub_avg4, file = 'SVM.csv', row.names = F)


