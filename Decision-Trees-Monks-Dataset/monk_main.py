#!/usr/bin/env python
###############################################################################################################
# Program : monks_main.py
# Author  : Kushal Kokje (kkokje@iu.edu), Udit Patel (udipatel@iu.edu)
# Version : 3.0
# Desc    : Program to build a multi value split Decision Tree to calculate accuracy of the monks dataset (UCI repo).
#           A learning curve is plotted with X-axis as depth of tree and Y-axis as accuracy
#           Program plots the confusion matrix for all the three monks test file for depth ranging from 0 to 10
#
# Change history :
#      Date             Version    Author          Description
#    09/10/2016          1.0      Kushal Kokje     Initial Version
#    09/13/2016          2.0      Udit Patel       Added indentation while printing output on the console and doc strings
#    09/23/2016          3.0      Kushal kokje     Final changes and added extra code comments.
###############################################################################################################

"""
 Plots the learning curve and confusion matrix for the monks dataset.
 NOTE : User will need to set the no_of_testfile parameter in the script
        equal to the number of test files that needs to be tested for a given training set
 Program usage : monks_main.py <str_filename> <int_treedepth>
                 for example : monks_main.py monks-1.train 8
 programs imports functions from 2 other python modules helper.py and decision_tree.py
"""

import sys
import re
import matplotlib.pyplot as pyplot
from decision_tree import *
from helper import *
import numpy as np

# set the parameter equal to the number of test files that needs to be tested for a given training set
####################################################################################################################
no_of_testfile = 3
#######################################################################################################################

print("=================== Run information ====================\n")
print("Program monks_main.py invoked and modules got imported successfully ")

if (len(sys.argv)!= 3):
    print("Enter the training data set file name and the depth of the tree in the given format : \n  monk_main.py <TRAINfilename> <int_depth>")
    sys.exit(0)
else:
    trainfile = sys.argv[1]
    treedepth = int(sys.argv[2])

print("parameters passed to the program are\n\t\t\t parameter 1 : %s \n\t\t\t parameter 2 : %d \n" %(trainfile,treedepth))

testfilename = trainfile.split('.')[0].split("-")[0] + "-"

# open the training file and read the features into the feature varaible
try:
    fin = open(trainfile,'r')
except IOError:
    print("ERROR : the train file %s couldn't be found" %trainfile)
    sys.exit(1)

# first line of the data file is list of features/attributes
features = fin.readline().split()

# define class attribute always set to "class"
class_attr = 'class'

# List having values starting from D0,D1,D2 .. till max depth of the tree and add Atestlale in the beginning
# Accuracy matrix is 2D table with columns as Depth and rows as accuracy for each test file
Acc_matrix_column = ["D" + str(d) for d in range(treedepth + 1)]
Acc_matrix_column.append("Atestlable")

print("Process started for training file %s " %trainfile )

line_count = 0
trainrecord = []

# read the train data file and print information like file name, attribute list,file count
for record in fin.readlines():
    line_count += 1
    trainrecord.append(re.sub(r'data.*$',"",record).strip())

print("Train filename : %s " %trainfile)
print("Classifier     : classic Decision Tree ID3 using Entropy and Information Gain")
print("Instances      : %d" %line_count)
print("Attributes     : %d " %len(features))

for ftr in features:
 print("                 %s" %ftr)

print("\nSTARTED : Decision Tree building and testing on Train set %s " %trainfile)

# zips the feature list and train data records together  like [{A:1,B:2},{A:3,B:4}] inside list[dict()] object
data = zipper(trainrecord,features)

# retrieves the majority class of the class attribute with the given data
major_val = majority_class(data, class_attr)
print("\t\t  Majority value for the class attribute is %s" %major_val)

# Initialize accuracy matrix
accuracy_matrix = {}

# temp list struct to store row of an accuracy
test_accuracy = []

# Iterate over all the test files to calculate accuracy for depth starting from 0 to maximun depth passed to the program
for fileindex in range(1,no_of_testfile + 1):

        # build the test file name on run time
        testfile = testfilename + str(fileindex) + ".test"

        # test_record  row lable for given test file
        test_record = "Atest" + str(fileindex)
        print("\n===> STARTED : Decision Tree building for test file  %s " %testfile)

        # open the test file and read the data into testlines list.
        try:
            fin2 = open(testfile,'r')
            next(fin2)
        except IOError:
            print(" ERROR : the file %s couldn't be found" %testfile)
            sys.exit(2)

        # print information of the test file like filename , attrbute list , no of rows
        testlines = []
        l_count = 0
        for line in fin2.readlines():
            l_count += 1
            testlines.append(re.sub(r'data.*$',"",line).strip())

        print(" \t\t Test filename  : %s " %testfile)
        print(" \t\t Classifier     : Classic Decision Tree ID3 using Entropy and Information Gain")
        print(" \t\t Instances      : %d" %l_count)
        print(" \t\t Attributes     : %d " %len(features))

        for ftr in features:
             print(" \t\t                  %s" %ftr)

        # zips the feature list and test data records together like [{A:1,B:2},{A:3,B:4}] inside list[dict()] object
        testdata = zipper(testlines,features)

        # iterate over tree depth fro 0 to max depth valuefor the give test file.
        for td in range(treedepth + 1):
            print("\n\t\t\t\t ==> STARTED : Process started for tree depth %d \n" %td)

            print("\t\t\t\t%%%%%%%%%%   Decision Tree    %%%%%%%%%%\n")

            # call the decision tree builder function to build the tree on train dataset for a given depth value
            dtree = build_decision_tree(data,features,class_attr,information_gain,td)
            print("\t\t\t\t %s " %str(dtree))
            print("\n")

            print("\t\t\t\t%%%%%%%%%%   Decision Tree    %%%%%%%%%%\n")

            print("\t\t\t\t STARTED : Classification process for test file %s " %testfile)

            # copy the test data lines to be classifed into classifyrecords list
            classifyrecords = testdata[:]

            # call to the classifier function that returns list of the predicted output
            predicated_output = classifier(dtree, classifyrecords)

            # Get the given class values from the test file and store into a list lst_output to calculate accuracy
            lst_output = []
            for record in testdata:
                lst_output.append(int(record[class_attr]))
            # print(lst_output)

            # Get the predicted class values and store into a list lst_predict to calculate accuracy
            lst_predict = []
            for item in predicated_output:
                # if a record is  nt classified , assign it the majority value of the given class
                if item is None:
                    lst_predict.append(int(major_val))
                else:
                    lst_predict.append(int(item))
            # print(predicated_output)

            # pass the given value and the predicted values list wto cal_accuracy function which returns accuracy
            # and other parameters like TP,TN,FP,FN
            parameters = calc_accuracy(lst_output,lst_predict)

            test_record = test_record + " " + str(round(parameters[0],3))
            print("\t\t\t\t ENDED   : Classification process for test file %s \n" %testfile)
            print("\t\t\t\t Accuracy of the model with Depth %d for file %s is %f%%  " %(td,testfile,round(parameters[0],3)))

            # call function  to plot the confusion matrix
            check = plot_confusionmatrix(parameters)

            # exit with error code 5 if unable to plot the confusion matrix
            if check != 0:
                print("ERROR : Error in plotting the confusion matrix")
                exit(5)

            print("\n\t\t\t\t ==> ENDED : Process ended for tree depth %d \n" %td)

        test_accuracy.append(test_record)
        fin2.close()
        print(" ===> ENDED : Decision Tree building for test file  %s \n\n " %testfile)
        print("printing test accuracy")
        print(test_accuracy)

Acc_matrix_column.sort()
print(Acc_matrix_column)

# print(Acc_matrix_column)
accuracy_matrix = zipper(test_accuracy,Acc_matrix_column)
print(accuracy_matrix)

# make a list with values D0 til D-max to plot on xaxis for plotting the learning curve
xaxis = ["D"+ str(x) for x in range(treedepth + 1)]
xax = range(treedepth + 1)
yaxis = []


# average over the values of all the test files and store the output in a list to plot on y -axis
for x in xaxis:
 temp = 0.0
 for attr in accuracy_matrix:
   temp += float(attr[x])
 yaxis.append(round((temp/no_of_testfile),3))

# convert list to Numpy arrays for plotting in Matplot lib
xax = np.array(xax)
yax = np.array(yaxis)

print("X-axis tree depth  values      : %s "  %str(xax))
print("Y-axis accuracy values         : %s \n" %str(yax))

print("Printing the learning curve \n ")
print("ENDED : Decision Tree building and testing on Train set %s " %trainfile)
pyplot.plot(xax,yax,linewidth=2.0)
pyplot.ylabel("Accuracy in %% ")
pyplot.xlabel("Depth of the tree")
pyplot.show()

fin.close()

exit(0)



















