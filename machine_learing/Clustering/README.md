# Clustering
* often you will find yourseftrying to create groups from data, instead of trying to predict classes or values
* you can think of it as an attempt to create labels
* you input some unlabled data, and the unsupervided learning algo retunes back possible clusters of the data
* you have data that only contains features and you want to see if there are patterns in the data that would allow you to create groups or clusters
* thius isa key distinction from our previous supervised learning tasks, where we hadhistorical labled data.
* we will have unlabled data, and attempt to 'discover' possible labels, through clustering
* by thenature of this problem, it can be difficultto evaluate the groups or clusters  for 'correctness'
* a large part of being able to interpret the clusters assined comes down to domain knowledge
* maybe you have some customer data, and then cluster them into distince groups
* it will be up to you to decide what the groups actually repressent
* sometimes this is easy, sometimes its really hard
* ex you could clustertumors into two groups, hoping to seperate between benign and malignat
* there is no guarantee that the clusters will fall along thoes lines, it will just split into the two most seperable groups 
* also depending on the clustering algo, it may be up to you to decide beforehand how many clusters you expect to create
* alot of clusteing problems have np 100% correct approach or answer, this is the nature of unsupervised learning
* lets continue by discussing k-means clustering

# reading
* Chapter 10 of Introduction to Statistical Learning by Gareth James, et al.

# k means clustering 
* is an unsuperivied learning  alog  that will attemp to group[ similar clusters together in your data
* the overall goal is to divide data into distinct groupss such that observations within each group are similar

## what do typical clustering problems look like
* clustering Similar documents
* cluster customers based on features
* market segmentation
* identify similar physocal groups

# the K Means Algo
* Choose a number of Clusters "K"
* randomly assign each point to a cluster

## Untill clusters stop changing, repeat the following
* for each cluster, compute the cluster centroid by taking the mean vector of points in the cluster
* Assign each data point to the cluster for which the centroid is the closest

# choosing a K Value
* there is no easy answer for choosing a "best" k  value 
## one way is the elbow method
* first  of all, compute the sum of squared error(SSE) for some values of k(for example2,4,6,8,etc.). 
* the SSE is defined as the sum of the squared  distance between each memeber of the cluster and its centoid.
* if you plot k against theSEE, you will see that the erro decreases as k gets larger: this si because when the number of clusters increase, they shoul besmalled, so distortion is also smaller.
* the idea of the elbow method is to choose the k at which the SSE decreases abruptly
* this produces an "elbow effect"
* dont take thsi as a strict rule when choosing a k value
* a lot of depends more on the context of the exact situation(domain  knowledge)

# pyspark
* by itself doesnt supprt a plotting mechanism, but you could use collect() and then plot the results with matplotlib or other visualization libaries

