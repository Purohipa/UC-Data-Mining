library(MASS)
data(Boston)
set.seed(12948181)
index <- sample(nrow(Boston),nrow(Boston)*0.75)
boston.train <- Boston[index,]
boston.test <- Boston[-index,]

summary(boston.train)
str(boston.train)
#Linear Regression######################################################################################
model_1 <- lm(medv~., data=boston.train)
summary(model_1)

model_AIC <- step(model_1)
model.AIC<-summary(model_AIC)
(model.AIC$sigma)^2
model.AIC$r.squared
model.AIC$adj.r.squared


#lm(formula = medv ~ crim + zn + chas + nox + rm + dis + rad + tax + ptratio + black + lstat, data = boston.train)

#Out of sample performance
model_2<- lm(formula = medv ~ crim + zn + chas + nox + rm + dis + rad + tax + ptratio + black + lstat, data = boston.test)
model_test<- summary(model_2)
(model_test$sigma)^2
model_test$r.squared
model_test$adj.r.squared


#Decision Trees##########################################################################################
library(rpart)
library(rpart.plot)

boston.largetree <- rpart(formula = medv ~ ., data = boston.train)
boston.largetree
prp(boston.largetree,digits = 4, extra = 1)
par(mfrow=c(1,2))
plotcp(boston.largetree)
printcp(boston.largetree)

boston.rpart <-prune(boston.largetree, cp = 0.05)
prp(boston.rpart,digits = 4, extra = 1)

boston.train.pred.tree = predict(boston.rpart)
boston.test.pred.tree = predict(boston.rpart,boston.test)
MSE.tree<- mean((boston.train.pred.tree - boston.train$medv)^2)
MSPE.tree <- mean((boston.test.pred.tree - boston.test$medv)^2)

#Bagging#############################################################################################
install.packages("ipred")
library(ipred)
boston.bag<- bagging(medv~., data = boston.train, nbagg=100)
boston.bag
boston.bag.train<- predict(boston.bag)
mean((boston.train$medv-boston.bag.train)^2)
boston.bag.pred<- predict(boston.bag, newdata = boston.test)
mean((boston.test$medv-boston.bag.pred)^2)

ntree<- c(1, 3, 5, seq(10, 200, 10))
MSE.test<- rep(0, length(ntree))
for(i in 1:length(ntree)){
  boston.bag1<- bagging(medv~., data = boston.train, nbagg=ntree[i])
  boston.bag.pred1<- predict(boston.bag1, newdata = boston.test)
  MSE.test[i]<- mean((boston.test$medv-boston.bag.pred1)^2)
}
par(mfrow=c(1,1))
plot(ntree, MSE.test, type = 'l', col=2, lwd=2, xaxt="n")
axis(1, at = ntree, las=1)

#out of bag 
boston.bag.oob<- bagging(medv~., data = boston.train, coob=T, nbagg=100)
boston.bag.oob

#Random Forest#####################################################################################
install.packages("randomForest")
library(randomForest)
boston.rf<- randomForest(medv~., data = boston.train, importance=TRUE)
boston.rf
boston.rf$importance
plot(boston.rf$mse, type='l', col=2, lwd=2, xlab = "ntree", ylab = "OOB Error")
boston.rf.pr<- predict(boston.rf)
mean((boston.train$medv-boston.rf.pr)^2)
boston.rf.pred<- predict(boston.rf, boston.test)
mean((boston.test$medv-boston.rf.pred)^2)
oob.err<- rep(0, 13)
test.err<- rep(0, 13)
for(i in 1:13){
  fit<- randomForest(medv~., data = boston.train, mtry=i)
  oob.err[i]<- fit$mse[500]
  test.err[i]<- mean((boston.test$medv-predict(fit, boston.test))^2)
  cat(i, " ")
}

matplot(cbind(test.err, oob.err), pch=15, col = c("red", "blue"), type = "b", ylab = "MSE", xlab = "mtry")
legend("topright", legend = c("test Error", "OOB Error"), pch = 15, col = c("red", "blue"))


#Boosting ####################################################################################
install.packages("gbm")
library(gbm)
boston.boost<- gbm(medv~., data = boston.train, distribution = "gaussian", n.trees = 10000, shrinkage = 0.01, interaction.depth = 8)
summary(boston.boost)

par(mfrow=c(1,1))
plot(boston.boost, i="lstat")
plot(boston.boost, i="rm")

boston.boost.pred.train<- predict(boston.boost,n.trees=10000)
mean((boston.train$medv-boston.boost.pred.train)^2)
boston.boost.pred.test<- predict(boston.boost, boston.test, n.trees = 10000)
mean((boston.test$medv-boston.boost.pred.test)^2)

ntree<- seq(100, 10000, 100)
predmat<- predict(boston.boost, newdata = boston.test, n.trees = ntree)
err<- apply((predmat-boston.test$medv)^2, 2, mean)
plot(ntree, err, type = 'l', col=2, lwd=2, xlab = "n.trees", ylab = "Test MSE")
abline(h=min(test.err), lty=2)

######GAM Models#########################################################################################
install.packages("mgcv")
library(mgcv)

boston.gam <- gam(formula= medv~s(crim)+s(zn)+s(indus)+chas+s(nox)+s(rm)+s(age)+s(dis)+rad+s(tax)+
                    s(ptratio)+s(black)+s(lstat), data=boston.train)
summary(boston.gam)
plot(boston.gam,shade=TRUE,seWithMean=TRUE,scale=0, pages = 1)
boston.gam.pred.train<- predict(boston.gam)
mean((boston.train$medv-boston.gam.pred.train)^2)
boston.gam.pred.test<- predict(boston.gam, boston.test)
mean((boston.test$medv-boston.gam.pred.test)^2)

#####Neural Net##########################################################################################
maxs <- apply(Boston, 2, max) 
mins <- apply(Boston, 2, min)

scaled <- as.data.frame(scale(Boston, center = mins, scale = maxs - mins))

train_ <- scaled[index,]
test_ <- scaled[-index,]
install.packages("neuralnet")
library(neuralnet)
n <- names(train_)
f <- as.formula(paste("medv ~", paste(n[!n %in% "medv"], collapse = " + ")))
nn <- neuralnet(f,data=train_,hidden=c(5,3),linear.output=T)
?neuralnet
plot(nn)
#MSE of training set
pr.nn <- compute(nn,train_[,1:13])
pr.nn_ <- pr.nn$net.result*(max(Boston$medv)-min(Boston$medv))+min(Boston$medv)
train.r <- (train_$medv)*(max(Boston$medv)-min(Boston$medv))+min(Boston$medv)

MSE.nn <- sum((train.r - pr.nn_)^2)/nrow(train_)
MSE.nn
# MSE of testing set
pr.nn <- compute(nn,test_[,1:13])
pr.nn_ <- pr.nn$net.result*(max(Boston$medv)-min(Boston$medv))+min(Boston$medv)
test.r <- (test_$medv)*(max(Boston$medv)-min(Boston$medv))+min(Boston$medv)

MSE.nn <- sum((test.r - pr.nn_)^2)/nrow(test_)
MSE.nn