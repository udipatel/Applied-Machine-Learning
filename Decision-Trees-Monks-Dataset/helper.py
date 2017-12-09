#!/usr/bin/env python
###############################################################################################################
# Program : helper.py
# Author  : Udit Patel (udipatel@iu.edu), Kushal Kokje (kkokje@iu.edu)
# Version : 3.0
# Desc    : Module includes helper functions like majority_class,get_values,plot_confusionmatrix,
#           zipper,subset_data and print_tree that are used to spit data set and print the confusion matriz
#
# Change history :
#      Date             Version    Author          Description
#   09/11/2016          1.0       Udit Patel       Initial version
#   09/19/2016          2.0       Kushal Kokje     Changed formulas for plotting confusion matrix
#   09/21/2016          3.0       Udit Patel       Added round function to round of fractions to 3
###############################################################################################################

def majority_class(inputdata,attr):
    """
    :param inputdata: Input data in the format  list(dict()) like [{A:1,B:2},{A:3,B:4}]
    :param attr: Attribute value for which we need to determine the majority class
    :return: returns the attribute value which has the maximum value
    """

    max_feature = None
    max_val = 0
    pdata = inputdata[:]
    class_lst = [record[attr] for record in pdata]

    # returns teh values with highest count in the data)
    set_lst = list(set(class_lst))

    for val in set_lst:
        if class_lst.count(val) > max_val:
            max_feature = val
            max_val = class_lst.count(val)
    return max_feature

def get_values(data, attr):
    """
    :param data: Input data in the format  list(dict()) like [{A:1,B:2},{A:3,B:4}]
    :param attr: attribute for which list of unique values needs to be returned
    :return: returns the list of unique values for the selected attribute
    """

    inputdata = data[:]
    return list(set([record[attr] for record in inputdata]))

def plot_confusionmatrix(parameters):
    """
    :param parameters: list having values [accuracy,true_positive,true,negative,false_positive,false_negative]
    :return: int 0 if the confusion matrix is plotted without any error
    """
    para = parameters[:]

    tp = para[1]
    tn = para[2]
    fp = para[3]
    fn = para[4]

    # calulation for Misclassification Rate , True positive rate (TPR) , True Negative rate (TNR)
    MR = round((fp + fn) / (tp + tn + fp + fn), 3) * 100
    TPR = round((tp) / (tp + fn), 3) * 100
    TNR = round((tn) / (tn + fp), 3) * 100

    print("\n")
    print("\t\t\t\t%%%%%%%%%%   Confusion Matrix    %%%%%%%%%%")
    print("\n")

    print(" \t\t\t\t A=0            B=1        <-------- predicted class ")

    print(" \t\t\t\t %d            %d         A = 0 " % (tn, fp))
    print(" \t\t\t\t %d            %d         B = 1 " % (fn, tp))
    print("\n")

    print("\t\t\t\t Misclassification Rate   = %f %%" % round(MR,3))
    print("\t\t\t\t True positive rate (TPR) = %f %%" % round(TPR,3))
    print("\t\t\t\t True Negative rate (TNR) = %f %% \n" % round(TNR,3))

    return 0


def zipper(lines, attributes):
    """
    :param lines: Input lines that are to be zipped with attribute values format is like ['1 2 3','4 5 6','7 8 9'].
    :param attributes: list of the attribute values format like ['A','B','C']
    :return: zipped list[dict()] like [{'A':1,'B':2,'c':3},{A':4,'B':5,'c':6},{A':7,'B':8,'c':9}]
    """

    zip_lines = lines
    zip_attr = attributes
    data = []
    for line in zip_lines:
        # uses zip confusion zip feature and data
        data_dict = dict(zip(zip_attr, [line_attr.strip() for line_attr in line.split()]))
        data.append(data_dict)
    return data

def subset_data(data, attr, value):
    """
    :param data: Input data in the format  list(dict()) like [{A:1,B:2},{A:3,B:4}]
    :param attr: attribute column which should be checked for passed value
    :param value: value which needs to be matched with attribute value
    :return :List of records from the input data <data> for for which the attribute  <attr>
             matches the given value.

    """

    data = data[:]
    output_lst = []

    for record in data:
        if record[attr] == value:
            output_lst.append(record)

    return output_lst
