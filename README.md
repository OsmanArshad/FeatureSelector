# FeatureSelector

## Description
Feature Selector finds the most useful combination of features to accurately classify an object. The program takes in a text file of instances of objects belonging to two or more classes, with values for features attributed to those instances.

The program can be instructed to use three different search algorithms to find the best feature combinations, forward search, backward search, and beam search.

Foward search works by starting with an empty set and then calculating the accuracy of every possible combination of features by adding one feature at a time, after which it returns the set it finds to be the most accurate.

Backward search works in a similar manner, but instead starts with a full set of all the features, and then calculates the accuracy of every possible combination of features by removing one feature at a time.

Beam search works as a modified forward search, where within the calculating of the accuracy of the set, beam search will drop the calculation of the accuracy if it sees it will not find an accuracy higher than the highest accuracy it has found so far. In this manner beam search spends more time among those sets it finds to be more promising. This has a significant impact on the time it takes to generate an answer, especially for a large data set.

The output shows the accuracy of all considered combination of features and how each algorithm progressed through the sets of features to arrive at it's answer.

## Output
