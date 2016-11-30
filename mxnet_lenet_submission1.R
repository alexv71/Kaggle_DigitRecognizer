
# Digit recognition for the Kaggle challenge http://www.kaggle.com/c/digit-recognizer
# Realized creating additional examples using rotation and shifting source dataset examples
# Used MxNet library (convolutional networks) and EBImage library to rotating images
# 11/19/2016 version

library(mxnet)

#=====================================================================================
# Load data
train <- read.csv('./input/train.csv', header=TRUE)
test <- read.csv('./input/test.csv', header=TRUE)

#=====================================================================================
# Preprocess data
train <- data.matrix(train)
test <- data.matrix(test)

train.x <- train[,-1]
train.y <- train[,1]

train.x <- train.x/255
test <- test/255
#=====================================================================================
# Shift train dataset on 2 pix (left, right, top, bottom) - create synthetic examples
col.left <- c(seq(1,784,28), seq(2,784,28))
col.right <- c(seq(27,784,28), seq(28,784,28))

train.x.l <- train.x
train.x.l[,1:782] <- train.x.l[,3:784]
train.x.l[, col.right] <- 0

train.x.r <- train.x
train.x.r[,3:784] <- train.x.r[,1:782]
train.x.r[, col.left] <- 0

train.x.t <- train.x
train.x.t[,1:728] <- train.x.t[,57:784]
train.x.t[, 729:784] <- 0

train.x.b <- train.x
train.x.b[,57:784] <- train.x.b[,1:728]
train.x.b[, 1:56] <- 0

train.x <- rbind(train.x, train.x.l, train.x.r, train.x.t, train.x.b)
rm(train.x.l); rm(train.x.r); rm(train.x.t); rm(train.x.b); rm(train); rm(col.left); rm(col.right)
train.y <- c(train.y, train.y, train.y, train.y, train.y)

#=====================================================================================
# Rotate +-16 degrees (only synthetic examples)
library(EBImage)
set.seed(5271)
for(i in 42001:nrow(train.x)) {
        train.x[i,] <- as.numeric(rotate(matrix(train.x[i,], 28, 28), 16-round(runif(1, 1, 32)), output.dim = c(28,28)))
}

#=====================================================================================
# Shuffle dataset
set.seed(5271)
indx <- sample.int(nrow(train.x), nrow(train.x), replace = FALSE)
train.x <- train.x[indx, ]
train.y <- train.y[indx]

#=====================================================================================
# Final preprocessing data for MXNET
train.array <- t(train.x)
dim(train.array) <- c(28, 28, 1, nrow(train.x))
test.array <- t(test)
dim(test.array) <- c(28, 28, 1, nrow(test))

#=====================================================================================
# Create LeNet Network
# input
data <- mx.symbol.Variable('data')
# first conv
conv1 <- mx.symbol.Convolution(data=data, kernel=c(5,5), num_filter=20)
tanh1 <- mx.symbol.Activation(data=conv1, act_type="tanh")
pool1 <- mx.symbol.Pooling(data=tanh1, pool_type="max", kernel=c(2,2), stride=c(2,2))
# second conv
conv2 <- mx.symbol.Convolution(data=pool1, kernel=c(5,5), num_filter=50)
tanh2 <- mx.symbol.Activation(data=conv2, act_type="tanh")
pool2 <- mx.symbol.Pooling(data=tanh2, pool_type="max", kernel=c(2,2), stride=c(2,2))
# first fullc
flatten <- mx.symbol.Flatten(data=pool2)
fc1 <- mx.symbol.FullyConnected(data=flatten, num_hidden=500)
tanh3 <- mx.symbol.Activation(data=fc1, act_type="tanh")
# second fullc
fc2 <- mx.symbol.FullyConnected(data=tanh3, num_hidden=10)
# loss
lenet <- mx.symbol.SoftmaxOutput(data=fc2)

#=====================================================================================
# Train LeNet Network
device.cpu <- mx.cpu()
# device.cpu <- list(mx.cpu(0), mx.cpu(1))

mx.set.seed(5271)
tic <- proc.time()
model <- mx.model.FeedForward.create(lenet, X=train.array, y=train.y,
                                     ctx=device.cpu, num.round=15, array.batch.size=100,
                                     learning.rate=0.05, momentum=0.9, wd=0.00001,
                                     eval.metric=mx.metric.accuracy,
                                     epoch.end.callback=mx.callback.log.train.metric(100))
print(proc.time() - tic)

#=====================================================================================
# Make predictions and write submission file
preds <- predict(model, test.array)
pred.label <- max.col(t(preds)) - 1
submission <- data.frame(ImageId=1:nrow(test), Label=pred.label)
write.csv(submission, file='submission_mxnet.csv', row.names=FALSE, quote=FALSE)
