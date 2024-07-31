# K-Means Clustering 


This is a numpy implementation of K-means clustering. 

> What is k-means clustering?

Clustering is a technique used in unsupervised machine learning to group similar data points together based on their characteristics or features. It aims to find patterns or structures within the data without any prior knowledge or labels.

Clustering algorithms, such as K-means, work by iteratively assigning data points to clusters and updating the cluster centroids until convergence. The number of clusters, k, needs to be specified beforehand. The algorithm minimizes the within-cluster sum of squares, aiming to create compact and well-separated clusters.


> How is k-means implemented?

**Training**

1. Assign K, the number of clusters.
2. Randomly assign centeroids, which the data points will be clustered around.
3. Measure the euclidian distance from each datapoint to a centroid, and assgn the centroid with the shortest distance. 

$$d(p, q) = \sqrt{\sum\limits_{i=1}^{n} (q_i - p_i)^2}$$

4. Adjust the centroids to so they are at the mean of all the data points assigned. 
5. Repeat step 3 and 4 until there is no change in the value of centroids or max iters has reached.

**Prediction**

