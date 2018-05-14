# Recommender System
* reading = Recommender Systems by Jannach and Zanker
* fully developer and deployed recommendation systems can be complex and resource inensive
* keep this in mind as we continue, youll usually want someone with previous experience implementing a production recommendation system
* since  full recommender systems require a heavy linear algebra background we will try to provide only a high level overview in this lecture
* check out the book mentioned iun the beginning of the lecture for a deeper look into this topic

## two most common types if recommender systems:
* Content-Based(CB)
* Collaborative Filtering(CF)
* CF is more commonly used than CB because it usually gives better resuts and is relatively easy to understand(from an overall implementation perspective)
### Collaborative Filtering
* produces recommendations based on the knowledge of users attitue to items, that is it uses the "wisom of the crowd" to recommend items
### Content-based
* focus on the attributes of the items and give you recommendations based on the similarity between them

## the algorithm
* has the ability to do feature learning on its own, which means that it can start  to learn for itself what features to use
* these techniques aim to fill in the missing entries of a user-item association matrix 

## spark details
* spark.ml currently supports model-based collaborative filtering, in which users and products are described by a small set of latent factors that can be used to predict missing entries
* spark.ml uses the alternating least squares (ALS) algorithm to learn these latent factors
* your data needs to be in a specific format to work with Sparks ALS Recommendation Algorithm

## Alternating Least Squares (ALS) 
* basically a matrix Factorization approach to implement a recommendation algo you decompose your larger user/item matrix into lower dimensional user factors and item factors

## intuitive understanding of a recommender system
* imagine we have 3 customers: 1,2,3
* also have some movies: A, B, C
* Customers 1 and 2 really enjpy movies A and B and rate them 5 out of 5 stars
* #1 and #2 dislike movie C, and  give it a one star rating
* now we have a new customer #3 who reports a 5 star review for movie A 
* what new movies should we recommend B or C
* Well, based off collaborative filtering we recommend movie B, because Users #1 and #2 also enjoyed that (and movie A)

* a content based system would need to take Users into account
* it would just group movies together based off features(length, genre, actors, etc ...)
* Oftern real recommendation systems have combinations of methods

