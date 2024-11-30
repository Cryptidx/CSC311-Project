import numpy as np
from utils import load_train_csv, load_train_sparse, load_valid_csv, load_public_test_csv, evaluate
from knn import knn_impute_by_user_pred as knn_impute_by_user
from neural_network import evaluate_with_predictions as evaluate_nn
from item_response import evaluate_with_predictions as evaluate_irt
from item_response import irt
from neural_network import train, AutoEncoder
import torch.utils.data
import torch
from scipy.sparse import csr_matrix
from sklearn.utils import resample


basepath = "./data"
train_data = load_train_csv(basepath)
val_data = load_valid_csv(basepath)
test_data = load_public_test_csv(basepath)

# our best vals
# for knn and user version
k_knn = 11

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


def resample_students(data, num_bootstraps, sample_size=None):
    """
    Resample students while keeping questions consistent across bootstrapped groups.

    :param data: Original training data dictionary {user_id, question_id, is_correct}.
    :param num_bootstraps: Number of bootstrapped datasets to generate.
    :param sample_size: Number of students to sample. Defaults to the total number of unique users.
    :return: List of resampled training data dictionaries.
    """
    # Get unique user IDs
    unique_users = list(set(data["user_id"]))

    # Default to sampling all users
    if sample_size is None:
        sample_size = len(unique_users)

    # Prepare the bootstrapped datasets
    bootstrapped_datasets = []

    for _ in range(num_bootstraps):
        # Resample users with replacement
        resampled_users = resample(unique_users, n_samples=sample_size, replace=True)

        # Collect all rows corresponding to the resampled users
        resampled_data = {"user_id": [], "question_id": [], "is_correct": []}
        for user in resampled_users:
            # Find all rows for the current user
            indices = [i for i, u in enumerate(data["user_id"]) if u == user]
            # Append the user's responses to the resampled data
            for idx in indices:
                resampled_data["user_id"].append(data["user_id"][idx])
                resampled_data["question_id"].append(data["question_id"][idx])
                resampled_data["is_correct"].append(data["is_correct"][idx])

        bootstrapped_datasets.append(resampled_data)

    return bootstrapped_datasets


def extract_validation_predictions(matrix, valid_data):
    """Extract predictions for the validation set from the full user-item matrix."""
    predictions = []
    for user_id, question_id in zip(valid_data["user_id"], valid_data["question_id"]):
        predictions.append(matrix[user_id, question_id])
    return np.array(predictions)


def resample_sparse_matrix(matrix, n_bootstraps):
    """
    Resample the rows of a sparse matrix with replacement.

    :param matrix: 2D sparse matrix (csr_matrix or ndarray)
    :param n_bootstraps: Number of bootstrap samples to generate.
    :return: List of resampled matrices.
    """
    # Ensure the input is a sparse CSR matrix
    if not isinstance(matrix, csr_matrix):
        matrix = csr_matrix(matrix)

    num_users = matrix.shape[0]  # Number of rows (users)
    bootstrapped_matrices = []

    for _ in range(n_bootstraps):
        # Sample row indices with replacement
        sampled_indices = np.random.choice(num_users, num_users, replace=True)

        # Resample rows to create a new sparse matrix
        resampled_matrix = matrix[sampled_indices, :]
        bootstrapped_matrices.append(resampled_matrix)

    return bootstrapped_matrices


def main():
    """
        ensemble process
    """

    np.random.seed(21)

    # create bootstraps for knn and nn
    sparse = load_train_sparse(basepath)
    bootstraps_n_knn = resample_sparse_matrix(sparse, 2)
    # create for irt
    bootstrap_irt = resample_students(train_data, 1)

    # irt turn
    theta, beta, val_acc_lst, train_lld_lst, val_lld_lst = irt(
        bootstrap_irt[0], val_data, lr_irt, iterations
    )
    pred_irt, val_acc_irt = evaluate_irt(val_data, theta, beta)
    pred_irt_test, val_acc_irt_test = evaluate_irt(test_data, theta, beta)

    # nn turn
    zero_train_matrix, train_matrix = load_data_nn(bootstraps_n_knn[0].toarray())
    model = AutoEncoder(zero_train_matrix.shape[1], k_nn)
    train(model, lr_net, lamb, train_matrix, zero_train_matrix, val_data, num_epoch)
    pred_nn, val_acc_nn = evaluate_nn(model, zero_train_matrix, val_data)
    pred_nn_test, val_acc_nn_test = evaluate_nn(model, zero_train_matrix, test_data)

    # knn turn
    pred_knn, val_acc_knn = knn_impute_by_user(bootstraps_n_knn[1].toarray(), val_data, k_knn)
    pred_knn_test, val_acc_knn_test = knn_impute_by_user(bootstraps_n_knn[1].toarray(), test_data, k_knn)

    # pred_knn = pred_knn.flatten()
    print("KNN SHAPE", pred_knn.shape)
    print("NN LENGTH", len(pred_nn))
    print("IRT LENGTH", len(pred_irt))

    # we want the format to match with irt and nn so we can bag it
    true_knn = extract_validation_predictions(pred_knn, val_data)
    true_knn_test = extract_validation_predictions(pred_knn_test, test_data)

    # figuring out the weights now just by trial and error based on
    # each parts' validation accuracy

    print("Knn val accuracy: ", val_acc_knn)
    print("nn val accuracy: ", val_acc_nn)
    print("irt val accuracy: ", val_acc_irt)

    predictions = np.array([true_knn, pred_irt, pred_nn])

    aggregated = np.mean(predictions, axis=0)
    validation_acc = evaluate(val_data, aggregated)
    print("Final validation accuracy is ", round(validation_acc, 3))

    # now for test
    predictions_test = np.array([true_knn_test, pred_irt_test, pred_nn_test])
    aggregated_test = np.mean(predictions_test, axis=0)
    test_acc = evaluate(test_data, aggregated_test)
    print("Final test accuracy is ", round(test_acc, 3))


if __name__ == "__main__":
    main()
