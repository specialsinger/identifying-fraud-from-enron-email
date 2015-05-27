#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.cross_validation import StratifiedShuffleSplit

## functions

# compute fraction is used for creating the new features
def computeFraction( poi_messages, all_messages ):
    ### in case of poi_messages or all_messages having "NaN" value, return 0.
    
    if poi_messages == "NaN" or all_messages =="NaN" :
        fraction = 0.0
    else:
        fraction = float(poi_messages) / float(all_messages)
        
    return fraction

PERF_FORMAT_STRING = "\
\tAccuracy: {:>0.{display_precision}f}\tPrecision: {:>0.{display_precision}f}\t\
Recall: {:>0.{display_precision}f}\tF1: {:>0.{display_precision}f}\tF2: {:>0.{display_precision}f}"
RESULTS_FORMAT_STRING = "\tTotal predictions: {:4d}\tTrue positives: {:4d}\tFalse positives: {:4d}\tFalse negatives: {:4d}\tTrue negatives: {:4d}"

# test_clssifier_withscale is using min max scaler to scale the features first
# and then using shuffle split to cross validate the classifier performance

def test_classifier_withscale(clf, dataset, feature_list, folds = 1000):
    data = featureFormat(dataset, feature_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)

    #scale the features  
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    features = scaler.fit_transform(features)
    
    cv = StratifiedShuffleSplit(labels, folds, random_state = 42)
    true_negatives = 0
    false_negatives = 0
    true_positives = 0
    false_positives = 0
    for train_idx, test_idx in cv: 
        features_train = []
        features_test  = []
        labels_train   = []
        labels_test    = []
        for ii in train_idx:
            features_train.append( features[ii] )
            labels_train.append( labels[ii] )
        for jj in test_idx:
            features_test.append( features[jj] )
            labels_test.append( labels[jj] )
        
        ### fit the classifier using training set, and test on test set
        clf.fit(features_train, labels_train)
        predictions = clf.predict(features_test)
        for prediction, truth in zip(predictions, labels_test):
            if prediction == 0 and truth == 0:
                true_negatives += 1
            elif prediction == 0 and truth == 1:
                false_negatives += 1
            elif prediction == 1 and truth == 0:
                false_positives += 1
            else:
                true_positives += 1
    total_predictions = true_negatives + false_negatives + false_positives + true_positives
    accuracy = 1.0*(true_positives + true_negatives)/total_predictions
    print '\n Scale the features and use shuffle split to validate the classifier'
    print clf
    print RESULTS_FORMAT_STRING.format(total_predictions, true_positives, false_positives, false_negatives, true_negatives)
    print accuracy

    try:
        total_predictions = true_negatives + false_negatives + false_positives + true_positives
        accuracy = 1.0*(true_positives + true_negatives)/total_predictions
        precision = 1.0*true_positives/(true_positives+false_positives)
        recall = 1.0*true_positives/(true_positives+false_negatives)
        f1 = 2.0 * true_positives/(2*true_positives + false_positives+false_negatives)
        f2 = (1+2.0*2.0) * precision*recall/(4*precision + recall)
        print PERF_FORMAT_STRING.format(accuracy, precision, recall, f1, f2, display_precision = 5)
        print RESULTS_FORMAT_STRING.format(total_predictions, true_positives, false_positives, false_negatives, true_negatives)
        print ""
    except:
        print "Got a divide by zero when trying out:", clf

# Use K-fold to validate the classifier
def test_classifier_Kfold(clf, dataset, feature_list, folds = 3):

    from sklearn import cross_validation
    from sklearn import datasets

    #features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels, test_size=0.5, random_state=42)

    kf = cross_validation.KFold(len(labels),folds)

    true_negatives = 0
    false_negatives = 0
    true_positives = 0
    false_positives = 0

    for train_indices, test_indices in kf:
        #make the training & tesint datasets
        features_train = [features[ii] for ii in train_indices]
        features_test = [features[ii] for ii in test_indices]
        labels_train = [labels[ii] for ii in train_indices]
        labels_test = [labels[ii] for ii in test_indices]
        clf.fit(features_train, labels_train)
        pred = clf.predict(features_test)


        from sklearn.metrics import accuracy_score
        acc = accuracy_score(pred, labels_test)
        print "accuracy_score", acc

        from sklearn.metrics import precision_score
        pre = precision_score(labels_test, pred)
        print "precision_score", pre

        from sklearn.metrics import recall_score
        recall = recall_score(labels_test, pred)
        print "recall_score", recall

        for prediction, truth in zip(pred, labels_test):
            if prediction == 0 and truth == 0:
                true_negatives += 1
            elif prediction == 0 and truth == 1:
                false_negatives += 1
            elif prediction == 1 and truth == 0:
                false_positives += 1
            else:
                true_positives += 1

    print '\n Use K-fold to validate the classfiier'
    print clf
    total_predictions = true_negatives + false_negatives + false_positives + true_positives
    print RESULTS_FORMAT_STRING.format(total_predictions, true_positives, false_positives, false_negatives, true_negatives)
    accuracy = 1.0*(true_positives + true_negatives)/total_predictions
    print "Accuracy: ", accuracy

    try:
        total_predictions = true_negatives + false_negatives + false_positives + true_positives
        accuracy = 1.0*(true_positives + true_negatives)/total_predictions
        precision = 1.0*true_positives/(true_positives+false_positives)
        recall = 1.0*true_positives/(true_positives+false_negatives)
        f1 = 2.0 * true_positives/(2*true_positives + false_positives+false_negatives)
        f2 = (1+2.0*2.0) * precision*recall/(4*precision + recall)
        print PERF_FORMAT_STRING.format(accuracy, precision, recall, f1, f2, display_precision = 5)
        print RESULTS_FORMAT_STRING.format(total_predictions, true_positives, false_positives, false_negatives, true_negatives)
        print ""
    except:
        print "Got a divide by zero when trying out:", clf


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary'] # You will need to use more features

### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )


# Data Explore
# compute & list the key characteristocs of this dataset
print "Dataset Summary\n"
print "total number of data points:", len(data_dict)
print "total number of features:", len(data_dict["SKILLING JEFFREY K"])

total_num = len(data_dict)

poi_num = 0
nemail_num = 0
nsalary_num = 0
nstock_num = 0

for name in data_dict:
    if data_dict[name]['poi'] == True:
        poi_num = poi_num + 1
    if data_dict[name]['email_address'] == 'NaN':
        nemail_num = nemail_num + 1
    if data_dict[name]['salary'] == 'NaN':
        nsalary_num = nsalary_num + 1
    if data_dict[name]['exercised_stock_options']=='NaN':
        nstock_num = nstock_num + 1

print "total number of POI:", poi_num
print "total number of non-POI:", total_num - poi_num
print "number of people have a known email:", total_num  - nemail_num
print "number of people have a quantified salary:", total_num - nsalary_num
print "number of people have a quantified exercised_stock_options:", total_num - nstock_num
print "\n"


### Task 2: Remove outliers

'''
# Analyze the financial data
features = ["salary", "bonus"]
data = featureFormat(data_dict, features)

# Draw the scatter plot
import matplotlib.pyplot

for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()
'''

# identified the outlier and remove from the dataset
data_dict.pop("TOTAL",0)

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

'''
# Draw the scatter plot of 'from_poi_to_this_person' and 'from_this_person_to_poi'

features = ['poi','from_poi_to_this_person', 'from_this_person_to_poi']
data = featureFormat(data_dict, features)

import matplotlib.pyplot

for point in data:
    from_poi = point[1]
    to_poi = point[2]
    if point[0] == 1:
        matplotlib.pyplot.scatter( from_poi, to_poi,color = "r", label="poi" )
    else:
        matplotlib.pyplot.scatter( from_poi, to_poi,color = "b", label="non-poi" )

matplotlib.pyplot.xlabel("from_poi_to_this_person")
matplotlib.pyplot.ylabel("from_this_person_to_poi")
matplotlib.pyplot.show()
'''

# create two new feature fraction_from_poi and fraction_to_poi
for name in my_dataset:
    data_point = my_dataset[name]
    from_poi_to_this_person = data_point["from_poi_to_this_person"]
    to_messages = data_point["to_messages"]
    fraction_from_poi = computeFraction( from_poi_to_this_person, to_messages )

    from_this_person_to_poi = data_point["from_this_person_to_poi"]
    from_messages = data_point["from_messages"]
    fraction_to_poi = computeFraction( from_this_person_to_poi, from_messages )

    my_dataset[name]['fraction_from_poi'] = fraction_from_poi
    my_dataset[name]['fraction_to_poi'] = fraction_to_poi
    
'''

# Draw the scatter plot of 'fraction_from_poi' and 'fraction_to_poi'

features = ['poi','fraction_from_poi', 'fraction_to_poi']
data = featureFormat(data_dict, features)

import matplotlib.pyplot

for point in data:
    from_poi = point[1]
    to_poi = point[2]
    if point[0] == 1:
        matplotlib.pyplot.scatter( from_poi, to_poi,color = "r", label="poi" )
    else:
        matplotlib.pyplot.scatter( from_poi, to_poi,color = "b", label="non-poi" )

matplotlib.pyplot.xlabel("fraction_from_poi")
matplotlib.pyplot.ylabel("fraction_to_poi")
matplotlib.pyplot.show()

'''

## Feature selection

# Univariate feature selection
import numpy as np

from sklearn.feature_selection import SelectPercentile, f_classif

all_features = ['poi', 'to_messages', 'deferral_payments', 'expenses','deferred_income','long_term_incentive', 'fraction_from_poi', 'restricted_stock_deferred', 'shared_receipt_with_poi', 'loan_advances', 'from_messages', 'other', 'director_fees', 'bonus', 'total_stock_value', 'from_poi_to_this_person', 'from_this_person_to_poi', 'restricted_stock', 'salary', 'total_payments', 'fraction_to_poi', 'exercised_stock_options']

testdata = featureFormat(my_dataset, all_features, sort_keys = True)
labels, features = targetFeatureSplit(testdata)
selector = SelectPercentile(f_classif, percentile=50)
selector.fit(features, labels)

featurenames = ['to_messages', 'deferral_payments', 'expenses','deferred_income','long_term_incentive', 'fraction_from_poi', 'restricted_stock_deferred', 'shared_receipt_with_poi', 'loan_advances', 'from_messages', 'other', 'director_fees', 'bonus', 'total_stock_value', 'from_poi_to_this_person', 'from_this_person_to_poi', 'restricted_stock', 'salary', 'total_payments', 'fraction_to_poi', 'exercised_stock_options']
print '\n Use univariate feature selection to rank all the features \n'
print '***Features sorted by score:', [(featurenames[i], selector.scores_[i]) for i in np.argsort(selector.scores_)[::-1]]
print '\n\n'

## Finalized the right number of features after manual testing 
features_list = ['poi','exercised_stock_options', 'total_stock_value', 'bonus', 'salary', 'fraction_to_poi','deferred_income', 'long_term_incentive']

#features_list = ['poi','exercised_stock_options']


### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)



### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()    # Provided to give you a starting point. Try a varity of classifiers


## try different algorithm

#from sklearn import tree
#clf = tree.DecisionTreeClassifier()

#from sklearn.svm import SVC
#clf = SVC()


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script.
### Because of the small size of the dataset, the script uses stratified
### shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

'''
## Test different validation method
# using K fold to validate the algorithm
test_classifier_Kfold(clf, my_dataset, features_list)

# feature scaled before using shuffle splict cross-validation, specific for SVC
test_classifier_withscale(clf, my_dataset, features_list) 
'''

test_classifier(clf, my_dataset, features_list)


### Dump your classifier, dataset, and features_list so 
### anyone can run/check your results.

dump_classifier_and_data(clf, my_dataset, features_list)
