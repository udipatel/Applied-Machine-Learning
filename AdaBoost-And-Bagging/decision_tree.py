#!/usr/bin/env python
###############################################################################################################
# Program : decision_tree.py
# Author  : Kushal Kokje (kkokje@iu.edu), Udit Patel (udipatel@iu.edu)
# Version : 2.0
# Desc    : Module includes functions related to optimization of ID3 algorithm such as entropy,information_gain,
#           get_bestattribute,classifier,get_prediction,build_decision_tree,calc_accuracy which are referred
#           in monk_main.py
#
# Change history :
#      Date             Version    Author           Description
#     09/11/2016        1.0       Kushal Kokje      Initial version
#     09/17/2016        2.0       Kushal Kokje      Edited build_decision_tree and entropy functions
#     09/19/2016        2.0       Udit Patel        Made changes in information gain function
###############################################################################################################

"""
  module included below mentioned functions :
  entropy,information_gain,get_bestattribute,classifier,get_predictedval,build_decision_tree,calc_accuracy
  Please use doctring  _doc_  and the function name for more details.
              print <functio_name>._doc_
"""
import math
from helper import *


def entropy(data, class_attr):
    """
    :param data       : Input data in the format  list(dict()) like [{A:1,B:2},{A:3,B:4}]
    :param class_attr : string with the value of the class attribute
    :return           : Entropy of the input data wrt to class attribute
    """
    dataval_freq = {}
    data_entropy = 0

    # Create  a list of all the values in the column class_attr in the given data
    data_values = [record[class_attr] for record in data]

    # create a dict storig the unique values in class and its count. kind of histogram
    classval_freq = {record: data_values.count(record) for record in data_values}
    len_data = len(data)

    # iterate through all the unique values and calculate the combined data entropy
    for val in classval_freq.values():
        data_entropy += (-val / len_data) * math.log(val / len_data, 2)

    return data_entropy


def information_gain(data, attribute, class_attr):
    """
    :param data      :  Input data in the format  list(dict()) like [{A:1,B:2},{A:3,B:4}]
    :param attribute :  Attribute on which to split the data to calculate information gain
    :param class_attr:  Class attribute of the input dataset
    :return          :  Final information gain wrt to given attribute
    """

    # Create  a list of all the values in the column attr in the given data
    class_values = [record[attribute] for record in data]  # changed

    # create historgram of the unique values and store the same in a dict() object
    classval_freq = {record: class_values.count(record) for record in class_values}

    total_val = sum(classval_freq.values())
    temp_entropy = 0.0

    # calculate the temp entropy of the subset
    for key in classval_freq.keys():
        val_prob = classval_freq[key] / total_val
        temp_data = [record for record in data if record[attribute] == key]
        temp_entropy += val_prob * entropy(temp_data, class_attr)

    # calculate information gain by subtratcing the main data entropy by the subset entropy. Uses recursion .
    info_gain = entropy(data, class_attr) - temp_entropy

    # retunrs the final info gain for the data parameter
    return (info_gain)

 # start code changes
 # for weighted entropy and information gain caculation data : 11/2/2016

def weighted_entropy(data, class_attr):
    """
    :param data       : Input data in the format  list(dict()) like [{A:1,B:2},{A:3,B:4}]
    :param class_attr : string with the value of the class attribute
    :return           : Entropy of the input data wrt to class attribute
    """
    dataval_freq = {}
    data_entropy = 0

    # print(" function weighted entropy called ")
    # Create  a list of all the values in the column class_attr in the given data
    data_values = [record[class_attr] for record in data]

    # create a dict storing the unique values in class and its count. kind of histogram
    classval_freq = {record: data_values.count(record) for record in data_values}
    weight_dict = {}
    for i in classval_freq.keys():
        total = 0
        for j in data:
            if j[class_attr] == i:
               total += j['weight']
    weight_dict[i] = total

    total_weight = sum(weight_dict.values())

    # iterate through all the unique values and calculate the combined data entropy
    for val in weight_dict.values():
        data_entropy += (-val / total_weight) * math.log(val / total_weight, 2)

    return data_entropy

def weighted_info_gain(data, attribute, class_attr):
    """
    :param data      :  Input data in the format  list(dict()) like [{A:1,B:2},{A:3,B:4}]
    :param attribute :  Attribute on which to split the data to calculate information gain
    :param class_attr:  Class attribute of the input dataset
    :return          :  Final information gain wrt to given attribute
    """

    # print(" function weighted info gain called ")
    # Create  a list of all the values in the column attr in the given data
    class_values = [record[attribute] for record in data]  # changed

    # create historgram of the unique values and store the same in a dict() object
    classval_freq = {record: class_values.count(record) for record in class_values}

    weight_dict = {}
    # attribute = 'A'

    for i in classval_freq.keys():
        total = 0
        for j in data:
            if j[attribute] == i:
                total += j['weight']
        weight_dict[i] = total

    total_val = sum(weight_dict.values())
    temp_entropy = 0.0

    # calculate the temp entropy of the subset
    for key in classval_freq.keys():
        val_prob = weight_dict[key] / total_val
        temp_data = [record for record in data if record[attribute] == key]
        temp_entropy += val_prob * weighted_entropy(temp_data, class_attr)

    # calculate information gain by subtratcing the main data entropy by the subset entropy. Uses recursion .
    info_gain = weighted_entropy(data, class_attr) - temp_entropy

    # retunrs the final info gain for the data parameter
    return (info_gain)

 # end code changes
 # for weighted entropy and information gain caculation data : 11/2/2016

def get_bestattribute(data, attributes, class_attr, gain_function):
    """
    :param data         : Input data in the format  list(dict()) like [{A:1,B:2},{A:3,B:4}]
    :param attributes   : Attribute on which to split the data to calculate information gain
    :param class_attr   : Class attribute of the input dataset
    :param gain_function: pointer to the information gai function
    :return             : string variable having the fittest attribute value on which to split
                          the dataset to expand the tree
    """

    max_gain_attr = None
    max_gain = 0.0
    data = data[:]

    # iterate over the attribute list to find info gain and decide the best attribute
    for val in attributes:
        gain = gain_function(data, val, class_attr)

        if (gain >= max_gain and val != class_attr):
            max_gain = gain
            max_gain_attr = val

    return max_gain_attr


def classifier(tree, data):
    """
    :param tree : dict() object holding the tree created by the build_decision_tree function
    :param data : Test data in the format  list(dict()) like [{A:1,B:2},{A:3,B:4}] for which
                  class value need to be predicted
    :return     : list with the predicted values of the test records
    """

    data = data[:]
    prediction = []
    i = 1

    # iterate over all the records in the test data to find the class prediction
    for record in data:
        # print("record %d started " %i)
        prediction.append(get_prediction(record, tree))
        # print("record %d ended " %i)
        i += 1

    # return the list with all te predicted values of the passed test dataset
    return prediction


def get_prediction(record, tree):
    """
    :param record : single record of the test data to be classified in the format dict() like {A:1,B:2}
    :param tree   : dict() object tree
    :return       : string or the leaf node of the tree
    """

    # base case to exit the recursion process.
    # The code traverses through the tree till a leaf node is found and returns the leaf node value
    if type(tree) is type("string"):
        return tree

    # Traverse the tree recursively till a leaf node is found.
    else:
        attr = list(tree)[0]
        try:
            t = tree[attr][record[attr]]
            # print(t)
        except KeyError:
            # if a path is not found return None value
            # print("Inside Key - error")
            return None

    # recursively call the function till a leaf node is foud
    return get_prediction(record, t)


def build_decision_tree(data, attributes, class_attr, gain_function, treedepth):
    """
    :param data          :  Input TRAIN data in the format  list(dict()) like [{A:1,B:2},{A:3,B:4}]
    :param attributes    :  List of attributes in the train dataset
    :param class_attr    :  class attribute value of train dataset
    :param gain_function :  pointer to the information gain function
    :param treedepth     :  maximum depth of the decision tree
    :return              :  dict() object named tree_dict which holds the learned decision  tree
    """
    # Returns a new decision tree as per the parameter values passed.

    # make a copy of the data to avoid changes in the original dataset
    data = data[:]
    # print(data)

    # Get a list of class values in the class column in data
    class_vals = [record[class_attr] for record in data]

    # find the majority value in the class_attr column in the given dataset
    major_val = majority_class(data, class_attr)

    # tree = {}
    # tree depth variable
    td = treedepth
    # print(td)

    # If the dataset is empty or the attributes list is empty, return the
    # default value. When checking the attributes list for emptiness, we
    # need to subtract 1 to account for the target attribute.

    # base case 1 : if all the attributes have been checked agiant the datset then exit the loop and return tree vaue
    if (len(attributes) - 1) <= 0:
        return major_val
    # base case 2 : If the data list becomes empty during the iteration process
    elif not data:
        return major_val
    # base case 3 : if all the values in a class attribute have same values
    elif class_vals.count(class_vals[0]) == len(class_vals):
        return class_vals[0]
    # If the maximum tree depth is reached as per the parameter passed to the program
    elif td == 0:
        return major_val

    else:
        # derive the best attribute to classify
        best_attr = get_bestattribute(data, attributes, class_attr,
                                      gain_function)

        # Create a new decision tree/node with the best attribute and an empty
        # dictionary object--we'll fill that up next.
        tree_dict = {best_attr: {}}
        td -= 1

        # Create a new decision tree/sub-node for each of the values in the
        # best attribute field

        for val in get_values(data, best_attr):
            # Create a subtree for the current value under the "best" field
            node = build_decision_tree(
                subset_data(data, best_attr, val),
                [attr for attr in attributes if attr != best_attr],
                class_attr,
                gain_function,
                td
            )

            # Add the new subtree to the empty dictionary object in our new tree.
            tree_dict[best_attr][val] = node

    return tree_dict


def calc_accuracy(listA, listB):
    """

    :param listA : List object holding  GIVEN    class value of test dataset
    :param listB : List object holding PREDICTED class value of the test dataset
    :return      : List object with values [accuracy, true_positive, true_negative, false_positive, false_negative]
    """
    lst1 = listA[:]
    lst2 = listB[:]

    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    # compare the given and the predicted values in the test file and derive the accuracy test parameters
    # print(lst1)
    # print(lst2)
    for indx in range(len(lst1)):
        if lst1[indx] == lst2[indx] and lst1[indx] == 1:
            true_positive += 1
        elif lst1[indx] == 0 and lst2[indx] == 1:
            false_positive += 1
        elif lst1[indx] == 1 and lst2[indx] == 0:
            false_negative += 1
        elif lst1[indx] == lst2[indx] and lst1[indx] == 0:
            true_negative += 1
        else:
            print("ERROR in Value ")
            exit(3)

    accuracy = ((true_positive + true_negative) / len(lst1)) * 100

    # return accuracy parameters list
    return [accuracy, true_positive, true_negative, false_positive, false_negative]
