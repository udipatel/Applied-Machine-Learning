#!/usr/bin/env python
###############################################################################################################
# Program : monks_main.py
# Author  : Kushal Kokje (kkokje@iu.edu), Udit Patel (udipatel@iu.edu)
# Version : 5.0
# Desc    : Program to build a multi value split Decision Tree to calculate accuracy of the monks dataset (UCI repo).
#           A learning curve is plotted with X-axis as depth of tree and Y-axis as accuracy
#           Program plots the confusion matrix for all the three monks test file for depth ranging from 0 to 10.
#           Also supports Bagging and AdaBoosting.
#
# Change history :
#      Date             Version    Author          Description
#    09/10/2016          1.0      Kushal Kokje     Initial Version
#    09/13/2016          2.0      Udit Patel       Added indentation while printing output on the console and doc strings
#    09/23/2016          3.0      Kushal kokje     Final changes and added extra code comments.
#    10/01/2016          4.0      udit patel       Added bagging code
#    10/02/2016          5.0      Kushal Kokje     updated Boosting code
###############################################################################################################

"""
 Plots the learning curve and confusion matrix for the monks dataset.
 NOTE : User will need to set the no_of_testfile parameter in the script
        equal to the number of test files that needs to be tested for a given training set
 Program usage : monk_main.py # <ensemble_type>  <tree_depth>  <num_bags/trees>  <data_set_path>

                 for example : monks_main.py "bag" 5 10 "Path/to/Datafiles"
 programs imports functions from 2 other python modules helper.py and decision_tree.py
"""

import sys
import re
import random
from decision_tree import *
from helper import *

# print("=================== Run information ====================\n")
# print("Program monks_main.py invoked and modules got imported successfully ")


def load_data(datapath):
    """
    Function: load_data(datapath)
    datapath: (String) the location of the UCI mushroom data set directory in memory

    This function loads the data set. datapath points to a directory holding
    agaricuslepiotatest1.csv and agaricuslepiotatrain1.csv. The data from each file
    is loaded and returned. All attribute values are nomimal. 30% of the data points
    are missing a value for attribute 11 and instead have a value of "?". For the
    purpose of these models, all attributes and data points are retained. The "?"
    value is treated as its own attribute value.

    Two nested lists are returned. The first list represents the training set and
    the second list represents the test set.
    """
    datasetpath = datapath

    try:
        fin = open(datasetpath+"/"+trainfile,'r')
    except IOError:
        print("ERROR : the train file %s couldn't be found" %trainfile)
        sys.exit(1)

    # first line of the data file is list of features/attributes
    # features_lst = fin.readline().split()
    features_lst = fin.readline()
    features = [fea for fea in features_lst.split(",")]

    line_count = 0
    inputrecords = []
    for record in fin.readlines():
        # print(record)
        # exit(10)
        line_count += 1
        inputrecords.append(record)

    # print("Train filename : %s " %trainfile)
    # print("Classifier     : classic Decision Tree ID3 using Entropy and Information Gain")
    # print("Instances      : %d" %line_count)
    # print("Attributes     : %d " %len(features))


    # zips the feature list and train data records together  like [{A:1,B:2},{A:3,B:4}] inside list[dict()] object
    trainrecord = zipper(inputrecords,features)

    # code to remove bruises-column from the training data
    data = []
    for rec in trainrecord:
        del rec['bruises?-no']
        data.append(rec)


    # open the test file
    try:
        fin2 = open(datasetpath+"/"+testfilename,'r')
        next(fin2) # skip header
    except IOError:
        print(" ERROR : the file %s couldn't be found" %testfilename)
        sys.exit(2)

    # print information of the test file like filename , attribute list , no of rows
    inputtestdata = []
    l_count = 0
    for line in fin2.readlines():
        l_count += 1
        inputtestdata.append(line)

    # print(" \t\t Test filename  : %s " %testfilename)
    # print(" \t\t Classifier     : Classic Decision Tree ID3 using Entropy and Information Gain")
    # print(" \t\t Instances      : %d" %l_count)
    # print(" \t\t Attributes     : %d " %len(features))


    # zips the feature list and test data records together like [{A:1,B:2},{A:3,B:4}] inside list[dict()] object
    testlines = zipper(inputtestdata,features)

    # remove column 'bruises?-no' from the test data
    testdata = []
    for rec in testlines:
        del rec['bruises?-no']
        testdata.append(rec)

    # remove column from features list
    features.remove('bruises?-no')

    td = tdepth

    train_test_data = []
    train_test_data.append(data)
    train_test_data.append(testdata)


    fin2.close()
    fin.close()

    return (train_test_data)


def learn_bagged(tdepth, numbags, datapath):
    """
    Function: learn_bagged(tdepth, numbags, datapath)
    tdepth: (Integer) depths to which to grow the decision trees
    numbags: (Integer)the number of bags to use to learn the trees
    datapath: (String) the location in memory where the data set is stored

    This function will manage coordinating the learning of the bagged ensemble.

    Nothing is returned, but the accuracy of the learned ensemble model is printed
    to the screen.
    """

    td = tdepth
    tottrees = numbags

    All_data = load_data(datapath)
    data = All_data[0]
    testdata = All_data[1]
    tottrees = nummodels

    line_count = len(data)
    l_count = len(testdata)
    features = list(data[0].keys())

    # Based on the esemble type value passed , execute the bagging or the boosting code.

    # print("Bagging code invoked")
    # create a empty list to store predicted o/p of all the tress geerated
    bag_list = []

    for randomtree in range(tottrees):
        # random selection of data based on choice
        rand_sample = list(map(lambda _: random.choice(data), range(line_count)))

        major_val = majority_class(rand_sample, class_attr)
        # print("\t\t  Majority value for the class attribute is %s" %major_val)

        # print("\n\t\t\t\t ==> STARTED : Process started for tree depth %d \n" %td)
        #
        # print("\t\t\t\t%%%%%%%%%%   Decision Tree    %%%%%%%%%%\n")

        dtree = build_decision_tree(rand_sample,features,class_attr,information_gain,td)
        # print("\t\t\t\t %s " %str(dtree))
        # print("\n")
        # exit(99)

        # print("\t\t\t\t%%%%%%%%%%   Decision Tree    %%%%%%%%%%\n")
        #
        # print("\t\t\t\t STARTED : Classification process for test file %s " %testfilename)

        # copy the test data lines to be classifed into classifyrecords list

        classifyrecords = testdata[:]
        # classifyrecords = list(map(lambda _: random.choice(testdata), range(20)))

        # call to the classifier function that returns list of the predicted output
        predicated_output = classifier(dtree, classifyrecords)


        cnt = 0
        for idx, item in enumerate(predicated_output):
            # if a record is  nt classified , assign it the majority value of the given class
            if item is None:
                predicated_output[idx]=int(major_val)
                cnt += 1
            else:
                predicated_output[idx] = int(item)

        bag_list.append(predicated_output)

        # End of the FOR loop

    from collections import Counter
    final_pred = []

    for i in range(l_count):
        l = []
        for idx,j in enumerate(bag_list):
            l.append(j[i])
        # print(l)
        most_common = Counter(l).most_common(1)[0][0]
        # print(most_common)
        final_pred.append(most_common)

    # Get the given class values from the test file and store into a list lst_output to calculate accuracy
    lst_output = []
    # print(len(testdata))
    for record in classifyrecords:  # to be changed to test data
        lst_output.append(int(record[class_attr]))
    # print(lst_output)

    # calculate TP,TN,FP,FN and accuracy
    parameters = calc_accuracy(lst_output,final_pred)

    # test_record = test_record + " " + str(round(parameters[0],3))
    # print("\t\t\t\t ENDED   : Classification process for test file %s \n" %testfilename)
    print("\t\t\t\t Accuracy of the model with Ensemble Type : bag ,Tree Depth : %d and total bags : %d for file %s is %f%%  " %(td,tottrees,testfilename,round(parameters[0],3)))

    # call function  to plot the confusion matrix
    check = plot_confusionmatrix(parameters)

    # exit with error code 5 if unable to plot the confusion matrix
    if check != 0:
        print("ERROR : Error in plotting the confusion matrix")
        exit(5)

    # End of Bagging code


def learn_boosted(tdepth, nummodels, datapath):
    """
    Function: learn_boosted(tdepth, numtrees, datapath)
    tdepth: (Integer) depths to which to grow the decision trees
    numtrees: (Integer) the number of boosted trees to learn
    datapath: (String) the location in memory where the data set is stored

    This function wil manage coordinating the learning of the boosted ensemble.

    Nothing is returned, but the accuracy of the learned ensemble model is printed
    to the screen.
    """
    tree_depth = tdepth
    total_iter = nummodels
    # start of the boosting code

    All_data = load_data(datapath)
    train_data = All_data[0]
    testdata = All_data[1]
    line_count = len(train_data)
    features = list(train_data[0].keys())

    # print("start of boosting code ==============================>>>>>>>>>>>>>>>>>>>> ")

    train_count = line_count
    # train_data = list(map(lambda _: random.choice(data), range(train_count)))

    initial_weight = 1/train_count
    weighted_data = []
    alpha = [1.0] * total_iter
    hypothesis = [None] * total_iter

    # add initial weights for each of the records in the training set
    for record in train_data:
        record['weight'] = initial_weight
        weighted_data.append(record)

    class_dis = [record[class_attr] for record in weighted_data]
    major_val = majority_class(weighted_data, class_attr)

    for i in range(total_iter):
        # print("\t\t  Majority value for the class attribute is %s" %major_val)
        # print("\n\t\t\t\t ==> STARTED : Process started for tree depth %d \n" %td)
        # print("\t\t\t\t%%%%%%%%%%  weighted Decision Tree    %%%%%%%%%%\n")

        # call the decision tree builder function to build the tree on train dataset for a given depth value
        wtree = build_decision_tree(weighted_data,features,class_attr,weighted_info_gain,tree_depth)
        hypothesis[i] = wtree
        # print("\t\t\t\t %s " %str(wtree))
        # print("\n")

        # print("\t\t\t\t%%%%%%%%%%  Weighted  Decision Tree    %%%%%%%%%%\n")
        #
        # print("\t\t\t\t STARTED : Classification process for train file %s " %trainfile)

        classifyrecords = weighted_data[:]
        # classifyrecords = list(map(lambda _: random.choice(testdata), range(20)))

        # call to the classifier function that returns list of the predicted output
        predicated_output = classifier(wtree, classifyrecords)
        predicated_output = [ int(x) for x in predicated_output ]

        weighted_output = []
        for record in weighted_data:
            weighted_output.append(int(record[class_attr]))

        # print((predicated_output))
        # print((weighted_output))

        parameters = calc_accuracy(weighted_output,predicated_output)


        # test_record = test_record + " " + str(round(parameters[0],3))

        # print("\t\t\t\t ENDED   : Classification process for test file %s \n" %testfilename)
        # print("\t\t\t\t Accuracy of the model with Depth %d for file %s is %f%%  " %(td,testfilename,round(parameters[0],3)))

        # calculate value of the error
        error = 1- round(parameters[0], 3)/100

        # print(" Error of the model is %f " % error)


        # call function  to plot the confusion matrix
        # check = plot_confusionmatrix(parameters)

        # exit with error code 5 if unable to plot the confusion matrix
        # if check != 0:
        #     print("ERROR : Error in plotting the confusion matrix")
        #     exit(5)

        # calculate alpha = 0.5 * math.log((1-error)/(error))
        alpha[i] = 0.5 * math.log((1-error)/(error))

        # print("alpha value of the model is %f" % alpha[i])


        # weighted_output,predicated_output

        # change class values from 0 to -1 for mathematical calculation simplicity

        for n,j in enumerate(weighted_output):
            if j == 0:
               weighted_output[n]= -1

        for n,j in enumerate(predicated_output):
            if j == 0:
               predicated_output[n]= -1


        # calculate the new weights based on the formuala

        for idx,rec in enumerate(weighted_output):
            weighted_data[idx]['weight'] = weighted_data[idx]['weight'] * math.exp( -alpha[i] * predicated_output[idx] * weighted_output[idx] )


        # Normalize the weights for a probalility distribution
        total_weight = 0.0

        # find the total weight
        for wt in weighted_data:
            total_weight += wt['weight']
        # divide all the weights with total_weight for normalization
        for idx,line in enumerate(weighted_data):
            weighted_data[idx]['weight'] = weighted_data[idx]['weight']/total_weight

         # updated weights will be passed in the next iterations
         # END of the FOR loop


    # Use the weak learners learned model to classify the test dataset
    classifyrecords = testdata[:]
    testrecords = []

    # Iterate thorugh the learned trees
    for idx,itree in enumerate(hypothesis):

        # Get the prediction for the current tree and convert to Int
        predicated_output = classifier(itree, classifyrecords)

        predicated_output = [ int(x.strip()) for x in predicated_output ]

        # change the 0 values in the predicted o/p to -1 for mathematical simplicity
        for n, j in enumerate(predicated_output):
            if j == 0:
                predicated_output[n] = -1

        # multily the predicted o/p of the weak classifier by the alpha value of the weak learner
        temp_rec = [rec * alpha[idx] for rec in predicated_output]

        # store the result of alpha*h(x) for a wek lerarner
        testrecords.append(temp_rec)

        # End of the FOR loop

    # Iterate through the records and predict the class based on sign function SIGN(sum(alpha*(h(x)))
    final_prediction = []

    for i in range(len(testdata)):
        l = []
        for idx,j in enumerate(testrecords):
            l.append(j[i])
        rec_sign = sum(l)
        if rec_sign <= 0:
            final_prediction.append(0)
        else:
            final_prediction.append(1)

    # Get the class o/p from the test file to compare with the predicted values

    tesfile_output = []
    for record in testdata:
        tesfile_output.append(int(record[class_attr]))

    # calulate TP,TN,FP,FN and accuracy
    parameters = calc_accuracy(tesfile_output,final_prediction)

    # print("\t\t\t\t ENDED   : Classification process for test file %s \n" %testfilename)
    # print("\t\t\t\t Accuracy of the model with Depth %d for file %s is %f%%  " %(td,testfilename,round(parameters[0],3)))
    print("\t\t\t\t Accuracy of the model with Ensemble Type : Boost, Depth %d and total Trees : %d for file %s is %f%%  " %(tree_depth,nummodels,testfilename,round(parameters[0],3)))
    # print("\n\t\t\t\t ==> ENDED : Process ended for tree depth %d \n" % td)

    check = plot_confusionmatrix(parameters)

        # exit with error code 5 if unable to plot the confusion matrix
    if check != 0:
        print("ERROR : Error in plotting the confusion matrix")
        exit(5)

    # end of the boosting code


if __name__ == "__main__" :

    if (len(sys.argv)!= 5):
        # print("Enter the training data set file name and the depth of the tree in the given format :"
        #       " \n  monk_main.py # <ensemble_type>  <tree_depth>  <num_bags/trees>  <data_set_path> ")
        sys.exit(99)
    else:
        entype     = sys.argv[1]
        tdepth     = int(sys.argv[2])
        nummodels  = int(sys.argv[3])
        datapath   = sys.argv[4]

    # set static parameters as required
    ####################################################################################################################
    trainfile    = "agaricuslepiotatrain1.csv"
    testfilename = "agaricuslepiotatest1.csv"
    class_attr   = 'bruises?-bruises'
    ####################################################################################################################

    # print("parameters passed to the program are \n\t\t\t parameter 1  : %s "
    #       "                                      \n\t\t\t parameter 2 : %d "
    #       "                                      \n\t\t\t parameter 3 : %d"
    #       "                                      \n\t\t\t parameter 4 : %s"
    #                                              %(ensemble_type,treedepth,tottrees,datasetpath)
    #        )

    # open the training file


    if entype == "bag":
        # Learned the bagged decision tree ensemble
        learn_bagged(tdepth, nummodels, datapath);
    else:
        # Learned the boosted decision tree ensemble
        learn_boosted(tdepth, nummodels, datapath);


    exit(0)





