#  Tree Methods
* a very powerful group of algos fall under the "Tree Methods" title
* decision trees, random forests, and gradient boosted trees
* chapter 8  of Introduction to Statistical learning By Gareth Jame s, et al.

# example of when to use a decision tree
*  Imagin that I play tenis every saturday and I always invite a friends to coime with me
* sometimes my friend shows up sometimes not
* for him it depend  on a variety of factors, such as:  weather, temp, humidity, wind etc..
* I start keeping track of these features and whether or not he showed up to play with me
* I want to use this data to predict whether or not he will show up to play
* an intuitive way to do this is through a Decision Tree
* in trees we have: Nodes, Edges, Root, Leaves
* Nodes: split for the value of a certain sttribue
* Edges: Outcome of a split to nesxt node
* Root: the node that preforms the first split
* Leaves: terminal nodes that predict the outcome

# how do we split on data when there is multiple varibles
* Entropy and Information Gain are the Mathmatical Methods of choosing the best split 

# Random Forests
* to imporve preformance, we canuse many trees with a random sample of features chosen as the split
* a new random sample of features is chose for every8 single tree at every single split
* works for both classification and regression tasks
* why to pick random? Suppose there is one very strong feature in the data set. Most of the trees will use that feature as the top split, resulting inan ensemble of similar trees that are highly correlated
*  averaging highly correlated quantities does not significantly reduce variance
* by randomly leaving out candidate feature form each split, Random Forests "decorrelates" the tree, such that the averaging process can reduce the variance of the resulting model

# Gradient boosting involves three  elements
*  a loss function to be optimized
* a weak learner to make predictions
* an additive model to add weak learners to minimize the loss function
##  Loss function
* a function in basic terms is the function/equatiuon you will use to determine how "far off" your predictions are
* wont have to deal with this using spark
## a weak learner
* decisiontrees are used as the weak learner in gradient boosting
* it is common to constrain the weak learners: such as a maximum numberof layers nodes, splits or leaf nodes.
# additive Model
* trees are added one at a time , and existing trees in the model are not changed
* a gradient descent procedure is used to minimize the loss when adding trees

# spark and grading boosting in 3 steps
* 1.) Train a weak model m using data samples drawn  according to some weight distribution
* 2.) Increase the weight of samples that are misclassified by modle m and decrease the weight of sample that are classified correctly by model m
* 3.) Train next weak model using samples drawn according to the updated weight distribution
* baically the algo always trains models using data samples that are "difficult" to learn in previous rounds, which results an ensemble of models that are good at learning different "parts" of the training data
* boosting weights of samples that were difficult to get correct
* the real detail of gradient boosting lies in the mathmaticsi
* if intrested in the mathmatics again look at chapter 8  of Introduction to Statistical learning By Gareth Jame s, et al.
* spark handles all of this under the hood for you so you can use thedefaults if you want, or dive into ISLR (book mentioned above) and begin to play around with the parameters


 

