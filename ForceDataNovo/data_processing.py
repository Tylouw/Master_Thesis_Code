import os
import csv
import sys
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
import random

from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.signal import savgol_filter

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix 
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier


def classify_list(values : list[list], labels: list[int]):
    """Given A list of list of values, classify them based on the items inside labels, and return them in two separate lists

    Args:
        values (list[list]): Contains values to separate
        labels (list[int]): values' labels

    Returns:
        tuple (list[list], list[list]): Return a Tuple containing list_negative and list_positive, which are values with labels 0 and 1
    """

    list_negative = [a for a, b in zip(values, labels) if b == 0]  # Values in A where B is 0
    list_positive = [a for a, b in zip(values, labels) if b == 1]  # Values in A where B is 1
    return list_negative, list_positive

def get_list_forces_folder(folderName: str, cut_start_time : float, cut_end_time : float, n_col : int):
    """Given the folderName, returns a list of list of forces of the given folder and a list of their labels

    Args:
        folderName (str): Path of the folder contianing time series data in csv format
        cut_start_time (float): Timestamp to start getting values
        cut_end_time (float): Timestamp to end getting values
        n_col (int): Number of column containing the data you want (you need to inspect the csv file first)

    Returns:
        tuple (list[list], list, list): Return a Tuple containing a list of data feature vector, their relative labels and their path
    """
    
    lista = []
    labels = []

    # Get all file names in the folder
    file_names = [f for f in os.listdir(folderName) if os.path.isfile(os.path.join(folderName, f))]
    # print("Printing the first file name: ", file_names[0])

    for i in range(len(file_names)):
        output_file = folderName + "/" + file_names[i]
        last_index = file_names[i].rfind('_')
        if(file_names[i][last_index + 1:] == "True.csv"):
            labels.append(1)
        else:
            labels.append(0) # Put zero by default

        # Check if the file exists
        if not os.path.exists(output_file):
            print("Error: You're trying to open an inexistent file")
            break

        force = []
        with open(output_file, mode='r') as file:
            reader = csv.reader(file)
            # Skip the header if the file has one
            next(reader, None)  

            for row in reader:
                if(float(row[0]) >= cut_start_time and float(row[0]) < cut_end_time):
                    force.append(float(row[n_col]))

        lista.append(force)

    return lista, labels, file_names

def get_all_forces_from_folders(folder_path : str, cut_start_time : float, cut_end_time : float, n_col : int, window_length : int, poly_order : int):
    # List all subfolders
    subfolders = [name for name in os.listdir(folder_path)
                if os.path.isdir(os.path.join(folder_path, name))]
    
    output_lista_all, input_labels_all, list_file_names_all = [], [], []
    for folder in subfolders:
        fp_tmp = os.path.join(folder_path, folder)
        output_lista, input_labels, list_file_names = get_feature_vector(fp_tmp, cut_start_time, cut_end_time, n_col, window_length, poly_order)
        output_lista_all.extend(output_lista)
        input_labels_all.extend(input_labels)
        list_file_names_all.extend(list_file_names)
    
    return output_lista_all, input_labels_all, list_file_names_all

def get_feature_vector(folderName: str, cut_start_time : float, cut_end_time : float, n_col : int, window_length : int, poly_order : int):
    """Given the folderName, returns a list of list of forces of the given folder with applied the SavGol Filter, and a list of their labels

    Args:
        folderName (str): Path of the folder contianing time series data in csv format
        cut_start_time (float): Timestamp to start getting values
        cut_end_time (float): Timestamp to end getting values
        n_col (int): Number of column containing the data you want (you need to inspect the csv file first)
        window_length (int): parameter for savgol preprocessing for windows length
        poly_order (int): parameter for savgol preprocessing for polynomial order

    Returns:
        tuple (list[list], list, list): Return a Tuple containing a list of data feature vector, their relative labels and their file path
    """
    input_lista, input_labels, list_file_names = get_list_forces_folder(folderName, cut_start_time, cut_end_time, n_col)
    output_lista = []

    for i in range(len(input_lista)):
        # Apply savgol_filter
        wl = window_length
        po = poly_order
        smoothed_force = savgol_filter(input_lista[i], window_length=wl, polyorder=po)
        output_lista.append(smoothed_force)

    return output_lista, input_labels, list_file_names

def create_threshold_sizes(k_neighbours: int):
    """Calculate all the percentage threshold given an integer

    Args:
        k_neighbours (int): k value of which we want the calculate 

    Returns:
        list: A list containing values in crescent oreder between [0.5, 1.0] representing the percentages of l-Value
    """
    # Apply different threshold and then plot them in a graph; Threshold = minimum number of neighbours that votes in favour of a certain class
    tmp_max = 1.0
    tmp_min = math.ceil(k_neighbours/2) / k_neighbours
    tmp_step = 1.0 / k_neighbours
    threshold_sizes = np.arange(tmp_min, tmp_max + 0.01, tmp_step).tolist()
    threshold_sizes = np.round(threshold_sizes, 3).tolist()

    return threshold_sizes

def get_mmm_values(folderName: str, cut_start_time: float, cut_end_time: float, n_col: int):
    """Calculate the max, mean and min values of all the data contained inside the folder

    Args:
        folderName (str): Path of the folder contianing time series data in csv format
        cut_start_time (float): Timestamp to start getting values
        cut_end_time (float): Timestamp to end getting values
        n_col (int): Number of column containing the data you want (you need to inspect the csv file first)

    Returns:
        tuple (list[list, list], list[list, list], list[list, list]): 
            Return a Tuple containing 3 list. Each list has two list, containing the max, min OR mean data, for Positive and Negative class
    """

    lista, labels = get_list_forces_folder(folderName, cut_start_time, cut_end_time, n_col)

    # (150, 1501)
    # Transpose the lista, so we obtain (1501, 150)
    transposed = [list(pair) for pair in zip(*lista)]
    
    # Max
    force_max_pos = []
    force_max_neg = []
    for i in range(len(transposed)):
        maxmax_pos = -sys.maxsize
        maxmax_neg = -sys.maxsize
        for j in range(len(transposed[i])):
            if(labels[j] == 1):
                if(transposed[i][j] > maxmax_pos):
                    maxmax_pos = transposed[i][j]
            else:
                if(transposed[i][j] > maxmax_neg):
                    maxmax_neg = transposed[i][j]
        if(maxmax_pos == -sys.maxsize):
            maxmax_pos = 0
        if(maxmax_neg == -sys.maxsize):
            maxmax_neg = 0
        force_max_pos.append(maxmax_pos)
        force_max_neg.append(maxmax_neg)

    # Min
    force_min_pos = []
    force_min_neg = []
    for i in range(len(transposed)):
        minmin_pos = sys.maxsize
        minmin_neg = sys.maxsize
        for j in range(len(transposed[i])):
            if(labels[j] == 1):
                if(transposed[i][j] < minmin_pos):
                    minmin_pos = transposed[i][j]
            else:
                if(transposed[i][j] < minmin_neg):
                    minmin_neg = transposed[i][j]
        if(minmin_pos == sys.maxsize):
            minmin_pos = 0
        if(minmin_neg == sys.maxsize):
            minmin_neg = 0
        force_min_pos.append(minmin_pos)
        force_min_neg.append(minmin_neg)

    # Mean
    force_mean_pos = []
    force_mean_neg = []
    for i in range(len(transposed)):
        sum_pos = 0.0
        sum_neg = 0.0
        pos = 0
        neg = 0
        for j in range(len(transposed[i])):
            if(labels[j] == 1):
                sum_pos += transposed[i][j]
                pos += 1
            else:
                sum_neg += transposed[i][j]
                neg += 1
        if(pos == 0): 
            pos = 1
        if(neg == 0):
            neg = 1
        force_mean_pos.append(sum_pos/float(pos))
        force_mean_neg.append(sum_neg/float(neg))
    
    return [force_max_pos, force_max_neg], [force_min_pos, force_min_neg], [force_mean_pos, force_mean_neg]

def get_list_values(csvFileName: str, ncol: int):
    """Get the values of the ncol of the csvFileName file

    Args:
        csvFileName (str): name of a .csv file
        ncol (int): Number of column containing the data you want (you need to inspect the csv file first)

    Returns:
        list: list containing the values of ncol of the csvFileName file
    """
    list_values = []
    with open(csvFileName, mode='r') as file:
        reader = csv.reader(file)
        # Skip the header if the file has one
        next(reader, None)  

        for row in reader:
            list_values.append(float(row[ncol]))

    return list_values

def modify_first_column(file_path: str, output_file_path: str):
    """This is used if you have file_path, whose first column of timestamps doesn't start from 0.0, 
    it substract to all the values in the first column the first value of the first column,
    and then save the modified file in a new file

    Args:
        file_path (str): Input file whose first column has to be modified
        output_file_path (str): Output file with the modifications
    """
    data = pd.read_csv(file_path)
    # Get the value from the first data row (second row overall) in the first column
    first_value = data.iloc[0, 0]
    # Subtract the first data row value from all rows in the first column (excluding the header)
    data.iloc[:, 0] = data.iloc[:, 0] - first_value
    # Write the modified data back to a new CSV file (or overwrite the original if you want)
    data.to_csv(output_file_path, index=False)
    print(f"Modified data saved to {output_file_path}")

def filter_hight_pass_data(data : list[list], threshold: float, labels=None):
    """Remoe from data (and labels) all the instances where the lists inside of data has at least a value >= threshold

    Args:
        data (list[list]): A list of list of values
        threshold (float): threshold value
        labels (list, optional): A list of associated labels to data. Defaults to None.

    Returns:
        tuple (list[list], list): Rturn a Tuple containing the original list without the found instances, and their relative labels
    """
    list_index_ge = []
    for i in range(len(data)):
        if any(x >= threshold for x in data[i]):
            list_index_ge.append(i)
    print("Index: ", list_index_ge)
    # Create a new list the do not contain the data in index list_index_ge
    result_data = [data[i] for i in range(len(data)) if i not in list_index_ge]
    if labels is None:
        result_labels = []
    else:
        result_labels = [labels[i] for i in range(len(data)) if i not in list_index_ge]
    return result_data, result_labels

def contains_ge_threshold(data : list, threshold: float):
    """Given a list, tells whether it contains at least one value >= threshold

    Args:
        data (list): List of values
        threshold (float): Threshold

    Returns:
        bool: 
    """
    if any(x >= threshold for x in data):
        return True
    else:
        return False
    
def get_overall_preformance(y_labels_pred: list[int], gt_labels: list[int]):
    """Calculate the evaluation metrics

    Args:
        y_labels_pred (list[int]): Prediction labels
        gt_labels (list[int]): Ground truth labels

    Returns:
        list: It contains in order: accuracy, precision, recall, f1_score, TP, FP, TN, FN
    """
    # Get indexes of non-"Uncertain" values
    indexes_certains_100 = [i for i, value in enumerate(y_labels_pred) if value != -1]
    certain = len(indexes_certains_100)
    # Extract values from gt anf y_pred based on indexes
    gt_labels_filtered = [gt_labels[i] for i in indexes_certains_100]
    y_pred_filtered = [y_labels_pred[i] for i in indexes_certains_100] 

    # Evaluation
    accuracy = accuracy_score(gt_labels_filtered, y_pred_filtered) * 100
    precision = precision_score(gt_labels_filtered, y_pred_filtered) * 100
    recall = recall_score(gt_labels_filtered, y_pred_filtered) * 100
    f1score = f1_score(gt_labels_filtered, y_pred_filtered) * 100

    cm = confusion_matrix(gt_labels_filtered, y_pred_filtered, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # print(f"Predicted Samples: {len(y_pred_filtered)} \nInitial Base Growing Dataset size: {train_frequency} \nBase Growing Dataset size: {len(X_dataset_train)} \nAccuracy: {accuracy:.2f}%\nPrecision: {precision:.2f}%\nRecall: {recall:.2f}%\nF1 Score: {f1score:.2f}%")
    # print("List of Uncertains: ", list_uncertain_iteration)
    # print("Train time total: " + str(sum(cost_train_time)) + " seconds")
    # print(f"TP = {tp}\nFP = {fp}\nTN = {tn}\nFN = {fn}")

    return accuracy, precision, recall, f1score, tp, fp, tn, fn, certain

def get_classification_indexes(y_true: list[int], y_pred: list[int]):
    """Find the indexes of TP, FP, TN, FN of the classification

    Args:
        y_true (list[int]): Ground truth labels
        y_pred (list[int]): Prediction labels

    Returns:
        tuple (list[int], list[int], list[int], list[int]):
            A tuple containing 4 list containing the indexes of TP, FP, TN and FN
    """
    tp = [i for i, (t, p) in enumerate(zip(y_true, y_pred)) if p != -1 and t == 1 and p == 1]
    fp = [i for i, (t, p) in enumerate(zip(y_true, y_pred)) if p != -1 and t == 0 and p == 1]
    tn = [i for i, (t, p) in enumerate(zip(y_true, y_pred)) if p != -1 and t == 0 and p == 0]
    fn = [i for i, (t, p) in enumerate(zip(y_true, y_pred)) if p != -1 and t == 1 and p == 0]
    return tp, fp, tn, fn

def performance_toString(accuracy: float, precision: float,  recall: float, f1score: float, 
                         tp: float, fp: float, tn: float, fn: float):
    """Print in the terminal the evaluation metric and the confusion matrix values

    Args:
        accuracy (float): Accuracy
        precision (float): Precision
        recall (float): Recall
        f1score (float): F1_Score
        tp (float): True Positivie
        fp (float): False Positive
        tn (float): True Negative
        fn (float): False Negative
    """
    print(f"Accuracy: {accuracy:.2f}%\nPrecision: {precision:.2f}%\nRecall: {recall:.2f}%\nF1 Score: {f1score:.2f}%")
    print(f"TP = {tp}\nFP = {fp}\nTN = {tn}\nFN = {fn}")

def reduce_feature_vector(positive: list[list], negative: list[list], reduce_factor: int):
    """Returns reduced feature np.array and corresponding numeric labels for positive and negative samples   

    Args:
        positive (list[list]): List of positive data values
        negative (list[list]): List of negative data values
        reduce_factor (int): Factor of which the original feature vector will be reduced with mean sliding window

    Returns:
        tuple (list[list], list[int]): list of all samples reduced and thei corresponding labels
    """

    X = []
    y = []
    for k in range(len(positive)):
        mean_positive = [float(np.mean(positive[k][i:i+reduce_factor])) for i in range(0, len(positive[k]), reduce_factor)]
        X.append(mean_positive)
        y.append("Positive")

    for k in range(len(negative)):
        mean_negative = [float(np.mean(negative[k][i:i+reduce_factor])) for i in range(0, len(negative[k]), reduce_factor)]
        X.append(mean_negative)
        y.append("Negative")

    X = np.array(X)
    
    encoder = LabelEncoder()
    print("Before encoder First Element: \n", y[0])
    y_numeric = encoder.fit_transform(y) # 1 = Pos, 0 = Neg
    print("After Encoder First Element: \n", y_numeric[0])

    return X, y_numeric

def train_val_test_split_scaled(X_tr : np.ndarray, y_tr, X_test : np.ndarray=None, y_test=None, train_size=0.8, random_state=40):
    """Split data into training and testing sets, and then fit them

    Args:
        X_tr (np.ndarray): Training dataset
        y_tr (_type_): Training labels
        X_test (np.ndarray, optional): Test dataset. Defaults to None.
        y_test (_type_, optional): Test labels. Defaults to None.
        train_size (float, optional): Training size ratio. Defaults to 0.8.
        random_state (int, optional): Random seed. Defaults to 40.

    Returns:
        tuple: In order, it will be returned the train, val and test dataset and labels
    """

    if not isinstance(y_tr, np.ndarray):
        y_tr = np.array(y_tr)  # Convert list to NumPy array
    
    if not isinstance(y_test, np.ndarray):
        y_test = np.array(y_test)  # Convert list to NumPy array

    scaler = StandardScaler()
    X_train, X_val, y_train, y_val = train_test_split(X_tr, y_tr, train_size=train_size, random_state=random_state)
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Handle X_test correctly
    if X_test is None:
        X_test_scaled = np.empty((0, X_train.shape[1]))  # Empty array with correct feature shape
    else:
        X_test_scaled = scaler.transform(X_test)

    # Handle y_test correctly
    if y_test is None:
        y_test = np.array([])

    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test

def compute_best_k(X, y_labels, min_neighbours=1, max_neighbours=50, plot=False):
    """Compute the best value of k of the current k-NN model

    Args:
        X (_type_): _description_
        y_labels (_type_): _description_
        min_neighbours (int, optional): _description_. Defaults to 1.
        max_neighbours (int, optional): _description_. Defaults to 50.
        plot (bool, optional): _description_. Defaults to False.

    Returns:
        tuple (int, float): _description_
    """

    # Normalize data to prevent data leakage, scale the features
    scaler = StandardScaler()
    X_k = scaler.fit_transform(X)
    print("The lenght of the Dataset is: " + str(len(X_k)))

    k_values = [i for i in range (min_neighbours, max_neighbours)]
    
    scores = []
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
        score = cross_val_score(knn, X_k, y_labels, cv=5)
        scores.append(np.mean(score))

    # Best K value and its score
    best_index = np.argmax(scores)
    best_k = k_values[best_index]
    best_score = scores[best_index]

    print("Best k = " + str(best_k) + " and its score = " + str(best_score))

    if plot:
        # Plot
        plt.xlabel("K Values with cross validation")
        plt.ylabel("Accuracy Score")
        plt.plot(k_values, scores)

    return best_k, best_score

def compute_evaluation_modified(y_true: list[int], y_pred: list[int]):
    """Compute Accuracy, Precision, Recall and F1_Score also considering the -1 class only in the denominator

    Args:
        y_true (list[int]): _description_
        y_pred (list[int]): _description_

    Returns:
        tuple (float, float, float, float): In order'they are Accuracy, Precision, Recall and F1_Score
    """

    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0

    for i in range(len(y_true)):
        if((y_pred[i] == y_true[i]) and y_pred[i] == 1):
            true_pos += 1
        elif((y_pred[i] == y_true[i]) and y_pred[i] == 0):
            true_neg += 1
        elif((y_pred[i] != y_true[i]) and y_true[i] == 1):
            false_neg += 1
        else:
            false_pos += 1
    
    # print(f"TP: {true_pos}, TN: {true_neg}, FP: {false_pos}, FN: {false_neg}")

    accuracy = np.nan if len(y_true) == 0 else 100 * (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)
    precision = 100.0 if (true_pos + false_pos) == 0 else 100 * (true_pos) / (true_pos + false_pos)
    recall = 100.0 if (true_pos + false_neg) == 0 else 100 * (true_pos) / (true_pos + false_neg)
    f1_score = 0.0 if (precision + recall) == 0 else 2*(precision*recall)/(precision + recall)

    # print(f"Evaluation Results:\n{" " :4}{"Accuracy" :<12} = {accuracy}\n{" " :4}{"Precision" :<12} = {precision}\n{" " :4}{"Recall" :<12} = {recall}\n{" " :4}{"F1Score" :<12} = {f1_score}")

    return accuracy, precision, recall, f1_score

def split_train_test(X_reduced_train: list[list], y_labels_train: list[int]):
    """Use this method if all your data is inside one folder and you want to randomly select 100 of them to keep as test only

    Args:
        X_reduced_train (list[list]): Dataset
        y_labels_train (list[int]): Labels

    Returns:
        tuple ( list[list], list[int], list[list], list[int]): The Dataset will be divided in Test and Train, and trie labels respectively
    """
    list_index_train = list(range(len(X_reduced_train)))  # Indexes

    index_test_tmp = random.sample(list_index_train, 100)  # Select 100 random elements

    X_reduced_test_tmp = [X_reduced_train[x] for x in index_test_tmp]  # The 100 elements
    y_labels_test_tmp = [y_labels_train[x] for x in index_test_tmp]

    X_reduced_train_tmp = [X_reduced_train[x] for x in list_index_train if x not in index_test_tmp]  # The remaining 604 elements
    y_labels_train_tmp = [y_labels_train[x] for x in list_index_train if x not in index_test_tmp] 
    return X_reduced_test_tmp, y_labels_test_tmp, X_reduced_train_tmp, y_labels_train_tmp







