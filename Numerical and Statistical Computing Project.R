# install tensorflow - tensorflow::install_tensorflow()
library(keras)
library(ggplot2)
library(datasets)
library(mclust)
library(ClusterR)
library(randomForest)

# Load dataset
mnist <- dataset_mnist()

# Set the desired subset sizes
train_size <- 6000
test_size <- 1000

# Subset the training data
train_indices <- sample(1:dim(mnist$train$x)[1], train_size, replace = FALSE)
X_train <- mnist$train$x[train_indices, , ]
y_train <- mnist$train$y[train_indices]

# Subset the test data
test_indices <- sample(1:dim(mnist$test$x)[1], test_size, replace = FALSE)
X_test <- mnist$test$x[test_indices, , ]
y_test <- mnist$test$y[test_indices]

# Check the dimensions of the subsets
dim(X_train)
length(y_train)
dim(X_test)
length(y_test)

# Reshape the input data into a vector
# We know each image has 28 x 28 = 784 pixels and each pixel is a variable
X_train <- matrix(X_train, ncol = 784)
X_test <- matrix(X_test, ncol = 784)
# Observe that the training set consists of 60000 images and test set consists
# of 10000 images
dim(X_train)
dim(X_test)

# For reference, to view the vector representing image 1 in the training set:
X_train[1,]

# Normalize input data
# Pixel strength values range from 0 to 255 so we divide all by 255
X_train <- X_train / 255
X_test <- X_test / 255

# Convert labels to categorical vectors with 10 classes (0 through 9)
to_categorical <- function(integer_labels, num_classes = NULL) {
  if (is.null(num_classes))
    num_classes <- max(integer_labels) + 1
  out <- matrix(0, nrow = length(integer_labels), ncol = num_classes)
  rows <- seq_along(integer_labels)
  cols <- integer_labels + 1
  out[cbind(rows, cols)] <- 1
  return(out)
}
y_train <- to_categorical(y_train, num_classes = 10)
y_test <- to_categorical(y_test, num_classes = 10)

##############################################################
# PCA

# Display original image(s) before PCA
I <- matrix(X_train[5, 784:1], nrow = 28, ncol = 28, byrow = T)
I <- apply(I, 2, rev)
image(I, col = grey(seq(1, 0, length = 784)))

# Remove constant or zero columns from the training and test data
non_zero_cols <- apply(X_train, 2, function(x) !all(x == 0))
X_train <- X_train[, non_zero_cols]
X_test <- X_test[, non_zero_cols]

pca <- prcomp(X_train)
X_train_pca <- predict(pca, X_train)
X_test_pca <- predict(pca, X_test)
# Now we have the transformed dataset

# We can use it for variable selection: select a subset of principle
# components that capture the most variance (show the most change
# between images) and only use that subset in the classification.
# This would reduce dimensionality and improve efficiency and
# performance of the classification

# pca$rotation shows the principle components PC1,...,PC784 in order
# of most impactful to least impactful
View(pca$rotation)

# Find the variance accounted for by each PC
var_explained <- pca$sdev^2 / sum(pca$sdev^2)
# Find the cumulative variance accounted for
cumulative_var <- cumsum(var_explained)
head(cumulative_var)
# So PC1 accounts for ~9% of the total variance, PC1 + PC2 account for
# ~17%, PC1 + PC2 + PC3 account for ~23%, and so on

# Plot the number of PCs (t) vs the cumulative variance
df <- data.frame(t = 1:length(cumulative_var), 
                 cumulative_var = cumulative_var)
ggplot(df, aes(x = t, y = cumulative_var)) + geom_line() + geom_point() +
  geom_hline(yintercept = 0.95, color = "red") +
  labs(title = "Cumulative Variance Covered by Principle Components", 
       x = "Number of Principle Components", 
       y = "Cumulative Variance")
# The red line represents 90% of the total variance of the data
# So we can use a relatively small amount of PCs and still have 90% of
# the total variance

t <- which(cumulative_var >= 0.95)[1]
t
# So the first t PCs account for 90.01% of the total variance
cumulative_var[t]

# Use only the first t PCs
Xt_train_pca <- X_train_pca[, 1:t]
Xt_test_pca <- X_test_pca[, 1:t]


######################
# Random Forest Classifier

# Convert y_train from a matrix to a vector of class labels
# Subtract 1 to convert from 1-indexed to 0-indexed
train_labels <- apply(y_train, 1, which.max) - 1
test_labels <- apply(y_test, 1, which.max) - 1
# Convert to factors
class_levels <- as.character(0:9)
y_train_factor <- factor(train_labels, levels = class_levels)
y_test_factor <- factor(test_labels, levels = class_levels)

# Fit model
rf_model <- randomForest(x = X_train, y = y_train_factor)
pred_y_test <- predict(rf_model, newdata = X_test)

accuracy <- sum(pred_y_test == test_labels) / length(test_labels)
accuracy

# Confusion matrix
conf_matrix <- table(Actual = test_labels, Predicted = pred_y_test)
conf_matrix
# Confusion matrix
confusion_matrix <- as.data.frame(table(pred_y_test, test_labels))
ggplot(data = confusion_matrix,
       mapping = aes(x = pred_y_test,
                     y = test_labels)) +
  geom_tile(aes(fill = Freq)) +
  geom_text(aes(label = sprintf("%1.0f", Freq)), vjust = 1) +
  scale_fill_gradient(low = "darkgrey",
                      high = "blue",
                      trans = "log") +
  labs(title = "Misclassification Plot of Random Forest",
       x = "GMM Predicted Labels", y = "True Labels")

# Print one of the decision trees
dt1 <- getTree(rf_model, k=1, labelVar = F)
View(dt1)

# Cumulative Variance of PCs vs Test Accuracy
accuracies <- numeric()
k <- 1
for (i in seq(0.1, 1, 0.05)) {
  t <- which(cumulative_var >= i)[1]
  Xt_train_pca <- X_train_pca[, 1:t]
  Xt_test_pca <- X_test_pca[, 1:t]
  
  rf_model <- randomForest(x = Xt_train_pca, y = y_train_factor)
  pred_y_test <- predict(rf_model, newdata = Xt_test_pca)
  accuracy <- sum(pred_y_test == test_labels) / length(test_labels)
  accuracies[k] <- accuracy
  
  k <- k+1
}

df <- data.frame(cbind(seq(0.1, 1, 0.05), accuracies))
ggplot(data = df, aes(x = df[,1], y = df[,2])) + geom_line() + geom_point()

# Number of PCs vs Test Accuracy
accuracies <- numeric()
k <- 1
tmax <- which(cumulative_var >= 0.99)[1]
tmin <- which(cumulative_var >= 0.1)[1]
for (i in seq(10, tmax, 10)) {
  Xt_train_pca <- X_train_pca[, 1:i]
  Xt_test_pca <- X_test_pca[, 1:i]
  
  rf_model <- randomForest(x = Xt_train_pca, y = y_train_factor)
  pred_y_test <- predict(rf_model, newdata = Xt_test_pca)
  accuracy <- sum(pred_y_test == test_labels) / length(test_labels)
  accuracies[k] <- accuracy
  
  k <- k+1
}
df <- data.frame(cbind(seq(10, tmax, 10), 
                       accuracies[1:length(seq(10, tmax, 10))]))
ggplot(data = df, aes(x = df[,1], y = df[,2])) + geom_line() + geom_point() + 
  labs(x = "Number of Principal Components", y = "Random Forest Accuracy")


#####################################
# K-means Mini-Batch Clustering

k <- 10  # Number of clusters
batch_size <- 100  # Batch size
num_init <- 10  # Number of initializations
max_iters <- 100  # Maximum number of iterations
initializer <- "kmeans++"  # Initialization method
# Number of iterations to continue after calculating the best 
# within-cluster sum of squared errors
early_stop_iter <- 10

# Train k-means model
kmeans_model <- MiniBatchKmeans(
  data = X_train,
  clusters = k,
  batch_size = batch_size,
  num_init = num_init,
  max_iters = max_iters,
  initializer = initializer,
  early_stop_iter = early_stop_iter,
  verbose = TRUE
)

# Get cluster assignments for the test set
cluster_assignments_test <- predict(kmeans_model, newdata = X_test, 
                                    fuzzy = FALSE)

# Convert cluster assignments to categorical labels
cluster_labels_test <- factor(cluster_assignments_test, levels = 0:9)

# Compare k-means clustering with true labels in the test set
conf_matrix_kmeans_test <- table(cluster_labels_test, test_labels)
conf_matrix_kmeans_test

# Calculate accuracy of k-means clustering on the test set
accuracy_kmeans_test <- sum(diag(conf_matrix_kmeans_test)) / 
  sum(conf_matrix_kmeans_test)
accuracy_kmeans_test

# Plot the confusion matrix
ggplot(data = as.data.frame(conf_matrix_kmeans_test),
       aes(x = cluster_labels_test, y = test_labels, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = sprintf("%1.0f", Freq)), vjust = 1) +
  scale_fill_gradient(low = "darkgrey", high = "blue", trans = "log") +
  labs(title = "Misclassification Plot of K-means Clustering",
       x = "Cluster Labels", y = "True Labels")


#####################################
# GMM

# Use mclust package

# Mclust function only worked with a maximum of 600 training samples
# Possibly could only handle so much data
train_size <- 600
test_size <- 100
train_indices <- sample(1:dim(mnist$train$x)[1], train_size, replace = FALSE)
X_train <- mnist$train$x[train_indices, , ]
y_train <- mnist$train$y[train_indices]
# Subset the test data
test_indices <- sample(1:dim(mnist$test$x)[1], test_size, replace = FALSE)
X_test <- mnist$test$x[test_indices, , ]
y_test <- mnist$test$y[test_indices]
dim(X_train)
dim(y_train)
dim(X_test)
dim(y_test)
# Convert data matrix into vector
X_train <- matrix(X_train, ncol = 784)
X_test <- matrix(X_test, ncol = 784)
dim(X_train)
dim(X_test)
# Normalize
X_train <- X_train / 255
X_test <- X_test / 255
# Convert labels to categorical vectors with 10 classes (0 through 9)
to_categorical <- function(integer_labels, num_classes = NULL) {
  if (is.null(num_classes))
    num_classes <- max(integer_labels) + 1
  out <- matrix(0, nrow = length(integer_labels), ncol = num_classes)
  rows <- seq_along(integer_labels)
  cols <- integer_labels + 1
  out[cbind(rows, cols)] <- 1
  return(out)
}
y_train <- to_categorical(y_train, num_classes = 10)
y_test <- to_categorical(y_test, num_classes = 10)
# Remove constant or zero columns from the training and test data
non_zero_cols <- apply(X_train, 2, function(x) !all(x == 0))
X_train <- X_train[, non_zero_cols]
X_test <- X_test[, non_zero_cols]

gmm_model <- Mclust(X_train, G = 10)
summary(gmm_model)

# Convert y_train from a matrix to a vector of class labels
# Subtract 1 to convert from 1-indexed to 0-indexed
true_labels <- apply(y_train, 1, which.max) - 1 

# Get the posterior probabilities for each observation
posterior_probs <- predict(gmm_model, newdata = X_train)$z

# Assign each observation to the cluster with the highest posterior probability
assigned_clusters <- apply(posterior_probs, 1, which.max)

# Map assigned clusters to digit labels based on majority voting
assigned_labels <- rep(0, nrow(X_train))
for (cluster in 1:10) {
  cluster_indices <- which(assigned_clusters == cluster)
  digit_labels <- true_labels[cluster_indices]
  assigned_labels[cluster_indices] <- 
    as.numeric(names(which.max(table(digit_labels))))
}

# Calculate accuracy
gmm_accuracy <- sum(assigned_labels == true_labels) / length(true_labels)

# Print assigned labels and accuracy
table(assigned_labels)
print(paste("Accuracy:", gmm_accuracy))

# Confusion matrix
assigned_labels <- factor(assigned_labels, levels = 0:9)
conf_matrix <- table(Actual = true_labels, 
                     Predicted = assigned_labels)
conf_matrix

confusion_matrix <- as.data.frame(table(assigned_labels, true_labels))
ggplot(data = confusion_matrix,
       mapping = aes(x = assigned_labels,
                     y = true_labels)) +
  geom_tile(aes(fill = Freq)) +
  geom_text(aes(label = sprintf("%1.0f", Freq)), vjust = 1) +
  scale_fill_gradient(low = "darkgrey",
                      high = "blue",
                      trans = "log") +
  labs(title = "Misclassification Plot of GMM",
       x = "GMM Predicted Labels", y = "True Labels")


####################################################
# Comparing GMM, Random Forest, and Kmeans Accuracies
# Random Forest Comparison
rf_comparison <- data.frame(
  Method = c("Random Forest"),
  Metric = c("Accuracy"),
  Value = c(accuracy)
)
# K-means Comparison
kmeans_comparison <- data.frame(
  Method = c("K-means"),
  Metric = c("Accuracy"),
  Value = c(accuracy_kmeans_test)
)
# GMM Comparison
gmm_comparison <-data.frame(
  Method = c("GMM"),
  Metric = c("Accuracy"),
  Value = c(gmm_accuracy)
)
# Combine both comparisons
comparison_data <- rbind(rf_comparison, kmeans_comparison, gmm_comparison)
# Plot
ggplot(comparison_data, aes(x = Method, y = Value, fill = Metric)) +
  geom_bar(stat = "identity", position = "dodge") +
  theme_minimal() +
  facet_wrap(~ Metric, scales = "free_y") +
  labs(
    title = "Comparison of Random Forest, K-means, GMM",
    x = "Method",
    y = "Value",
    fill = "Metric"
  ) +
  theme(legend.position = "bottom")
