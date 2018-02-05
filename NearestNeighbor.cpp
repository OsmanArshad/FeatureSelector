/*
    Title:  NearestNeighbor.cpp
    Author: Osman Arshad
    Email:  osmanaarshad@gmail.com
    Description: NearestNeighbor.cpp takes in a text file of "features", and by
    using one of three search algorithms, selects the releveant combination of 
    features in the data set, used for the classification of an instance of some object.
    
    Notes: Command for compile is: 
        g++ -o NearestNeighbor  NearestNeighbor.cpp
*/
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <vector>
#include <cmath>
using namespace std;

/*  Functions   */
void forwardSearch(vector<vector <double> > theData);
void backwardSearch(vector<vector <double> > theData);
void customSearch(vector< vector<double> > theData);
void normalizeData(vector <vector<double> > &theData);
void printFeatureSubset(vector<double> subset, int j);
bool intersect(vector<double> subset, int k);
double calculateEuclideanDistance(vector<double> x, vector<double> y, vector<double> subset, int p, string s);
double KFoldCrossValidation(vector <vector <double> > theData, vector<double> curFeats, int p, string searchType, int checkIfCustomAlgo);

// Take in a data set and standardize it
void normalizeData(vector <vector<double> > &theData)
{
    double mean, sum, sumSqrs, variance, stdDev;
    double numElements = theData.size();;

    for (int i = 1; i < theData[0].size(); i++)
    {
        sum = 0;
        sumSqrs = 0;
        mean = 0;
        variance = 0;
        stdDev = 0;

        for (int j = 0; j < numElements; j++)
        {
            sum += theData[j][i];
            sumSqrs += theData[j][i] * theData[j][i];
        }
        mean = sum / numElements;
        variance = (sumSqrs - ((sum * sum) / numElements)) / (numElements - 1);
        stdDev = sqrt(variance);

        for (int k = 0; k < theData.size(); k++)
            theData[k][i] = (theData[k][i] - mean) / stdDev;
    }
}

// Calculating the distance between two nieghbors in the data set
double calculateEuclideanDistance(vector<double> x, vector<double> y, vector<double> subset, int p, string s)
{
   double distance = 0;
   
   for (int i = 0; i < subset.size(); i++)
       distance += (pow(x[subset[i]] - y[subset[i]], 2));
   
   if (s == "forward")
        distance += (pow(x[p] - y[p], 2));
   return sqrt(distance);
}

// Print out the features collected by subset
void printFeatureSubset(vector<double> subset, int j = 0)
{
    cout << "{";
    if (j != 0) {   cout << j;   }
    for (int z = 0; z < subset.size(); z++)
    {
        if (j != 0) {   cout << ", "; j = 0;   }
        cout << subset[z];
        if (z != subset.size()-1)
        {
            cout << ", ";
        }
    }
    cout << "}";
}

// Check if feature k already exists in a subset
bool intersect(vector<double> subset, int k)
{
    for (int x = 0; x < subset.size(); x++)
    {
        if (subset[x] == k)
            return true;
    }
    return false;
}

// Remove a specific feature from a subset
vector <double> removeThisFeature(vector <double> subset, double x) 
{
	for (double i = 0; i < subset.size(); i++) 
    {
		if (subset.at(i) == x) 
        {
			subset.erase(subset.begin() + i);
			return subset;
		}
	}
	return subset;
}

// Calulating the K Fold Cross Validation of a dataset using a specified feature subset
// Extra parameters: searchType and checkIfCustomAlgo are used to determine which algorithm is calling this function
double KFoldCrossValidation(vector <vector <double> > theData, vector<double> curFeats, int p, string searchType, int checkIfCustomAlgo)
{
    vector<double> selectNeighbor;
    double correct = 0;
    double nearestNeighborDistance;
    double minNearestNeighborDistance;
    int fewestMissesSoFar = 100;
    int misses = 0;
    int isCustomAlgo = checkIfCustomAlgo;

    for (int i = 0; i < theData.size(); i++)
    {
        vector<double> leaveOneOutColumn = theData[i];
        minNearestNeighborDistance = 10000;

        for (int j = 0; j < theData.size(); j++)
        {
            if (i != j)
            {
                nearestNeighborDistance = calculateEuclideanDistance(leaveOneOutColumn, theData[j], curFeats, p, searchType);
                if (nearestNeighborDistance < minNearestNeighborDistance)
                {
                    minNearestNeighborDistance = nearestNeighborDistance;
                    selectNeighbor = theData[j];
                }
            }
        }
        if (selectNeighbor[0] == 1 && leaveOneOutColumn[0] == 1) {  correct++;  }
        else if (selectNeighbor[0] == 2 && leaveOneOutColumn[0] == 2)   {   correct++;  }
        else    {   misses++;   }

        if (isCustomAlgo == 1)
            if (misses > fewestMissesSoFar)  
                return 0;
    }
    fewestMissesSoFar = misses;
    return correct / (double)theData.size();
}

//  Forward search works by starting with an empty subset (current features), adding a single feature to it,
//  calculating the accuracy of the subset with that feature, and then repeating this process for all
//  combinations of features, until deciding on the most accurate feature subset
void forwardSearch(vector<vector <double> > theData)
{
    vector <double> currentFeatures;
    vector <double> bestFeatures;
    double alreadyAdded;
    double totalAccuracy = 0;
    double addFeature;
    double bestAccuracy = 0;
    double accuracy = 0;
    
    cout << "Beginning search.\n\n";
    for (int i = 1; i < theData[0].size(); i++)
    {
        bestAccuracy = 0;
        for (int j = 1; j < theData[0].size(); j++)
        {
            if (!intersect(currentFeatures, j))
            {
                accuracy = KFoldCrossValidation(theData, currentFeatures, j, "forward", 0);
                cout << "\tUsing feature(s) ";
                printFeatureSubset(currentFeatures, j);
                cout << " accuracy is " << accuracy * 100 << "%\n";
                if (accuracy > bestAccuracy)
                {
                    bestAccuracy = accuracy;
                    addFeature = j;
                }
            }
        }

        if (bestAccuracy < totalAccuracy)
        {
            cout << "\n{Warning, Accuracy has decreased! Continuing search ";
            cout << "in case of local maxima}";
        }

        if (alreadyAdded != addFeature)
        {
            currentFeatures.push_back(addFeature);
            cout << "\nFeature set ";
            printFeatureSubset(currentFeatures);
            cout << " was best, accuracy is " << bestAccuracy * 100 << "%\n\n";
        }
        alreadyAdded = addFeature;

        if (bestAccuracy > totalAccuracy)
        {
            totalAccuracy = bestAccuracy;
            bestFeatures = currentFeatures;
        }   
    }

    cout << "Finished search!! The best feature subset is: ";
    printFeatureSubset(bestFeatures);
    cout << " which has an accuracy of " << totalAccuracy * 100.0 << "%\n\n";
}

//  Backward search works by starting with a full subset containing all the features, removing a 
//  single feature from the subset, calculating the accuracy of the subset without that feature,
//  and then repeating this process for all combinations of features, until deciding on the most 
//  accurate feature subset
void backwardSearch(vector<vector <double> > theData)
{
    vector<double> bestFeatures;
    vector<double> currentFeatures;
    for (int p = 1; p < theData[0].size(); p++)
    {
        currentFeatures.push_back(p);
    }

    bestFeatures = currentFeatures;
    double accuracy = 0;
    double totalAccuracy = 0;
    double bestAccuracy = 0;
    double removeFeature;
    double alreadyRemoved;

    cout << "Beginning search.\n\n";
    for (int i = 1; i < theData[0].size() - 1; i++)
    {
        bestAccuracy = 0;
        for (int j = 1; j < theData[0].size(); j++)
        {
            if (intersect(currentFeatures, j))
            {
                vector<double> rmvdFeatures = removeThisFeature(currentFeatures, j);            
                accuracy = KFoldCrossValidation(theData, rmvdFeatures, j, "backward", 0);
                
                cout << "\tUsing feature(s) ";
                printFeatureSubset(rmvdFeatures);
                cout << " accuracy is " << accuracy * 100 << "%\n";
                
                if (accuracy > bestAccuracy)
                {
                    bestAccuracy = accuracy;
                    removeFeature = j;
                }               
            }
        }

        if (bestAccuracy < totalAccuracy)
        {
            cout << "\n{Warning, Accuracy has decreased! Continuing search ";
            cout << "in case of local maxima}";
        }

        currentFeatures = removeThisFeature(currentFeatures, removeFeature);
        cout << "\nFeature set ";
        printFeatureSubset(currentFeatures);
        cout << " was best, accuracy is " << bestAccuracy * 100 << "%\n\n";

        if (bestAccuracy > totalAccuracy)
        {
            totalAccuracy = bestAccuracy;
            bestFeatures = currentFeatures;
        }   
    }

    cout << "Finished search! The best feature subset is: ";
    printFeatureSubset(bestFeatures);
    cout << ", which has an accuracy of " << totalAccuracy * 100.0 << "%\n\n";
}

//  My custom search is a modified forward search algorithm, with the added feature of cutting off the
//  search algorithm early in its computation if it determines that further searching through the current
//  combination of features would not result in a higher accuracy
void customSearch(vector< vector<double> > theData)
{
    vector <double> currentFeatures;
    vector <double> bestFeatures;
    double alreadyAdded;
    double totalAccuracy = 0;
    double addFeature;
    double bestAccuracy = 0;
    double accuracy = 0;
    
    cout << "Beginning search.\n\n";
    for (int i = 1; i < theData[0].size(); i++)
    {
        bestAccuracy = 0;
        for (int j = 1; j < theData[0].size(); j++)
        {
            if (!intersect(currentFeatures, j))
            {
                accuracy = KFoldCrossValidation(theData, currentFeatures, j, "forward", 1);
                
                if (accuracy != 0)
                {
                    cout << "\tUsing feature(s) ";
                    printFeatureSubset(currentFeatures, j);
                    cout << " accuracy is " << accuracy * 100 << "%\n";
                }

                if (accuracy == 0)
                {
                    cout << "\tAccuracy of adding feature " << j << " is determined to be lower than ";
                    cout << "current best accuracy, so we skip calculating its accuracy.\n";
                }
                if (accuracy > bestAccuracy)
                {
                    bestAccuracy = accuracy;
                    addFeature = j;
                }
            }
        }

        if (bestAccuracy < totalAccuracy)
        {
            cout << "\n{Warning, Accuracy has decreased! Continuing search ";
            cout << "in case of local maxima}";
        }

        if (alreadyAdded != addFeature)
        {
            currentFeatures.push_back(addFeature);
            cout << "\nFeature set ";
            printFeatureSubset(currentFeatures);
            cout << " was best, accuracy is " << bestAccuracy * 100 << "%\n\n";
        }
        alreadyAdded = addFeature;

        if (bestAccuracy > totalAccuracy)
        {
            totalAccuracy = bestAccuracy;
            bestFeatures = currentFeatures;
        }   
    }
    cout << "Finished search! The best feature subset is: ";
    printFeatureSubset(bestFeatures);
    cout << " which has an accuracy of " << totalAccuracy * 100.0 << "%\n\n";
}

int main()
{
    string testFileName;
    string row;
    ifstream inputFile;
    int algNum;
    double data;
    vector< vector<double> > fileData;

    cout << "Welcome to Osman Arshad's Feature Selection Algorithm\n";
    cout << "Type in the name of the file to test: \n";
    cin >> testFileName;

    inputFile.open(testFileName.c_str());
    if (!inputFile) 
    {
        cout << "Unable to open file\n";
        return 1;
    }

    while (getline(inputFile, row)) 
    {
        stringstream rowValues(row);

        vector<double> rowData;
        while (rowValues >> data)
        {
            rowData.push_back(data);
        }
        fileData.push_back(rowData);
    }    
    inputFile.close();

    int numFeatures = fileData[0].size() - 1;
    int numInstances = fileData.size();

    cout << "Type the number of the algorithm you want to run.\n";
    cout << "\t 1) Forward Selection\n";
    cout << "\t 2) Backward Elimination\n";
    cout << "\t 3) Osman's Special Algorithm\n";
    cin >> algNum;

    cout << "This dataset has " << numFeatures;
    cout << " features (not including the class attribute),";
    cout << " with " << numInstances << " instances\n";
    cout << "\nPlease wait while I normalize the data...\n\n";

    normalizeData(fileData);

    if (algNum == 1) {  forwardSearch(fileData);    }
    if (algNum == 2) {  backwardSearch(fileData);   }
    if (algNum == 3) {  customSearch(fileData);    }

return 0;
}