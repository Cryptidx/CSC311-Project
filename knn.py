import numpy as np
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
from utils import (
    load_valid_csv,
    load_public_test_csv,
    load_train_sparse,
    sparse_matrix_evaluate,
)


def knn_impute_by_user(matrix, valid_data, k):
    """Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.

    See https://scikit-learn.org/stable/modules/generated/sklearn.
    impute.KNNImputer.html for details.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """

    # closest user , then it gets their answr and compares
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure
    mat = nbrs.fit_transform(matrix)
    acc = sparse_matrix_evaluate(valid_data, mat)
    print(f"Validation Accuracy (k == {k}): {round(acc,3)}")
    return acc


def knn_impute_by_item(matrix, valid_data, k):
    """Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """

    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure
    mat = nbrs.fit_transform(matrix.T)
    acc = sparse_matrix_evaluate(valid_data, mat.T)
    print(f"Validation Accuracy (k == {k}): {round(acc, 3)}")
    #####################################################################

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return acc


def main():
    sparse_matrix = load_train_sparse("./data").toarray()
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")

    print("Sparse matrix:")
    print(sparse_matrix)
    print("Shape of sparse matrix:")
    print(sparse_matrix.shape)

    print("#################################################")

    #####################################################################                                                      #
    # Compute the validation accuracy for each k. Then pick k* with     #
    # the best performance and report the test accuracy with the        #
    # chosen k*.                                                        #
    #####################################################################
    ks = [1, 6, 11, 16, 21, 26]
    validation_accuracies = []

    print("With User Based filtering:")
    for k in ks:
        acc = knn_impute_by_user(sparse_matrix, val_data, k)
        validation_accuracies.append(acc)

    # plot user based
    plt.plot(ks, validation_accuracies, label="User-Based")

    index = max(range(len(validation_accuracies)), key=lambda i: validation_accuracies[i])
    k = ks[index]

    print(f"I choose k == {k}")
    acc = knn_impute_by_user(sparse_matrix, test_data, k)
    print(f"Test accuracy (k == {k}): {round(acc,3)}")

    print("#################################################")
    print("With Item Based filtering:")
    validation_accuracies = []

    for k in ks:
        acc = knn_impute_by_item(sparse_matrix, val_data, k)
        validation_accuracies.append(acc)

    # plot item based
    plt.plot(ks, validation_accuracies, label="Item-Based")
    plt.xlabel("K values")
    plt.ylabel("Accuracy score")
    plt.title('Validation accuracy score based on different k values')
    plt.legend()
    plt.savefig("knn.png")

    index = max(range(len(validation_accuracies)), key=lambda i: validation_accuracies[i])
    k = ks[index]

    print(f"I choose k == {k}")
    acc = knn_impute_by_user(sparse_matrix, test_data, k)
    print(f"Test accuracy (k == {k}): {round(acc, 3)}")
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
