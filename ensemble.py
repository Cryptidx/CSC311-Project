# TODO: complete this file.
import numpy as np
import random
from utils import load_train_csv, load_valid_csv, load_public_test_csv,evaluate
from knn import knn_impute_by_user_pred as knn_impute_by_user
from neural_network import evaluate_with_predictions as evaluate_nn
from item_response import evaluate_with_predictions as evaluate_irt
from item_response import irt
from neural_network import train, AutoEncoder
from torch import FloatTensor
from scipy.sparse import coo_matrix
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torch

basepath = "./data"

train_data = load_train_csv(basepath)
val_data = load_valid_csv(basepath)
test_data = load_public_test_csv(basepath)


users = max(train_data["user_id"]) + 1
questions = max(train_data["question_id"]) + 1


# our best vals
# for knn and user version
k_knn = 5

# for irt
lr_irt = 0.01
iterations = 50


# and neurl nets
lr_net = 0.005
num_epoch = 65
lamb = 0.001
k_nn = 100

def load_data_nn(train_matrix):
    """Load the data in PyTorch Tensor.

    :return: (zero_train_matrix, train_data, valid_data, test_data)
        WHERE:
        zero_train_matrix: 2D sparse matrix where missing entries are
        filled with 0.
        train_data: 2D sparse matrix
        valid_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
        test_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
    """

    zero_train_matrix = train_matrix.copy()
    # Fill in the missing entries to 0.
    zero_train_matrix[np.isnan(train_matrix)] = 0
    # Change to Float Tensor for PyTorch.
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    train_matrix = torch.FloatTensor(train_matrix)

    return zero_train_matrix, train_matrix

def resample_data(data, sample_size=None):
    """
    Resample the training data with replacement.

    :param data: Original training data dictionary {user_id, question_id, is_correct}.
    :param sample_size: Number of samples to draw. Defaults to the size of the original data.
    :return: Resampled training data dictionary.
    """
    if sample_size is None:
        sample_size = len(data["is_correct"])
    #
    # indices = np.random.choice(len(data["is_correct"]), size=sample_size, replace=True)
    # resampled_data = {
    #     "user_id": [data["user_id"][i] for i in indices],
    #     "question_id": [data["question_id"][i] for i in indices],
    #     "is_correct": [data["is_correct"][i] for i in indices]
    # }

    max_user_id = max(data["user_id"])
    max_question_id = max(data["question_id"])

    indices = np.random.choice(len(data["is_correct"]), size=sample_size, replace=True)
    resampled_data = {
        "user_id": [data["user_id"][i] for i in indices if data["user_id"][i] <= max_user_id],
        "question_id": [data["question_id"][i] for i in indices if data["question_id"][i] <= max_question_id],
        "is_correct": [data["is_correct"][i] for i in indices]
    }

    return resampled_data


def aggregate_predictions(predictions_list, weights=None):
    """
    Aggregate predictions using a weighted average.

    :param predictions_list: List of predictions from models.
    :param weights: List of weights for each model. Defaults to equal weights.
    :return: Aggregated predictions.
    """
    if weights is None:
        weights = [1 / len(predictions_list)] * len(predictions_list)

    weights = np.array(weights)
    aggregated = np.average(predictions_list, axis=0, weights=weights)
    return aggregated

#
# def dictionary_to_sparse(data, num_users, num_questions):
#     """
#     Convert a dictionary to a sparse matrix.
#
#     :param data: Dictionary with "user_id", "question_id", "is_correct".
#     :param num_users: Total number of users.
#     :param num_questions: Total number of questions.
#     :return: Sparse matrix representation of the data.
#     """
#
#    # print(data)
#     rows = np.array(data["user_id"])
#     cols = np.array(data["question_id"])
#     values = np.array(data["is_correct"])
#
#     np.set_printoptions(threshold=np.inf)
#     #print(values)
#
#     sparse_matrix = coo_matrix((values, (rows, cols)), shape=(num_users, num_questions))
#     sparse_matrix.maximum(1)
#     # np.set_printoptions(threshold=np.inf)
#     print(sparse_matrix)
#     return sparse_matrix
#

def extract_validation_predictions(matrix, valid_data):
    """Extract predictions for the validation set from the full user-item matrix."""
    predictions = []
    for user_id, question_id in zip(valid_data["question_id"],valid_data["user_id"]):
        predictions.append(matrix[user_id, question_id])
    return np.array(predictions)

def dictionary_to_sparse(data, num_users, num_questions):
    """
    Convert a dictionary to a dense matrix using for loops.
    :param data: Dictionary with "user_id", "question_id", "is_correct".
    :param num_users: Total number of users.
    :param num_questions: Total number of questions.
    :return: Dense matrix representation of the data.
    """
    # Initialize a dense matrix with zeros
    matrix = np.zeros((num_users, num_questions), dtype=int)

    # Populate the matrix with values from the data
    for user, question, correct in zip(data["user_id"], data["question_id"], data["is_correct"]):
        # Set the value to 1 if the user answered correctly
        matrix[user, question] = 1 if correct == 1 else 0

    return matrix

resampled_data_knn = resample_data(train_data)
resampled_data_irt = resample_data(train_data)
resampled_data_nn = resample_data(train_data)



knn_sparse = dictionary_to_sparse(resampled_data_knn, users, questions)
nn_sparse = dictionary_to_sparse(resampled_data_nn, users, questions)

def main():
    """
        gh
    """
    zero_train_matrix, train_matrix = load_data_nn(nn_sparse)

    model = AutoEncoder(zero_train_matrix.shape[1], k_nn)
    train(model, lr_net, lamb, train_matrix, zero_train_matrix, val_data, num_epoch)

    #
    #
    # # predictions irt2
    theta, beta, val_acc_lst, train_lld_lst, val_lld_lst = irt(
        train_data, val_data, lr_irt, iterations
    )

    pred_knn, val_acc_knn = knn_impute_by_user(knn_sparse, val_data, k_knn)
    pred_irt, val_acc_irt = evaluate_irt(val_data, theta, beta)
    pred_knn = extract_validation_predictions(pred_knn, val_data)
    pred_nn, val_acc_nn = evaluate_nn(model, zero_train_matrix, val_data)



    print("Val accuracy knn")
    print(val_acc_knn)
    print("Val accuracy irt")
    print(val_acc_irt)
    print("Val accuracy nn")
    print(val_acc_nn)

    #print(len(pred_knn[0]))

    #print(len(pred_knn))

    final_predictions = (pred_knn + pred_nn + pred_irt) / 3

    final_labels = (final_predictions >= 0.5).astype(int)

    accuracy = np.mean(final_labels == val_data["is_correct"])
    print(f"Validation Accuracy of Ensemble: {accuracy}")





    # knn should have 1's 0's or nans (make them 0's)
    # howeverrr mine currently has 2's 3's thats not right

    # print("knn")
    # # print(type(pred_knn))
    # print(len(pred_knn))
    # print(len(pred_knn[0]))
    # print(pred_knn)

    #
    # print("irt")
    # print(type(pred_irt))
    # print(pred_irt)
    # print(pred_irt)
    # print(len(pred_irt))
    #
    # print("nn")
    # print(type(pred_nn))
    # print(len(pred_nn[0]))
    # print(len(pred_nn))
    # print(pred_nn)


    # knn returns <class 'numpy.ndarray'>
    # some 0's and such

    # irt returns a list of fals's and trues. just a long ahh list

    # nn returns a list of I think actual predictions?


    #predictions_list = [predictions_knn, predictions_irt, predictions_nn]
    #final_predictions = aggregate_predictions(predictions_list, weights=[0.5, 0.3, 0.2])



if __name__ == "__main__":
    main()
