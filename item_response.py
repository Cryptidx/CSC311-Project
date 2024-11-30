from utils import (
    load_train_csv,
    load_valid_csv,
    load_public_test_csv,
    load_train_sparse,
)
import numpy as np
from matplotlib import pyplot as plt # TODO allowed to add?


def sigmoid(x):
    """Apply sigmoid function."""
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    log_lklihood = 0.0
    for n in range(len(data["user_id"])):
        i = data["user_id"][n]
        j = data["question_id"][n]
        c = data["is_correct"][n]
        x = theta[i] - beta[j]
        p_ij = sigmoid(x)
        # update log likelihood based on the derived expression from (a)
        log_lklihood += c * np.log(p_ij) + (1 - c) * np.log(1 - p_ij)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def update_theta_beta(data, lr, theta, beta):
    """Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    # First initiallize the gradients
    theta_gradient = np.zeros_like(theta)
    beta_gradient = np.zeros_like(beta)

    for n in range(len(data["user_id"])):
        i = data["user_id"][n]
        j = data["question_id"][n]
        c = data["is_correct"][n]
        x = theta[i] - beta[j]
        p_ij = sigmoid(x)
        diff = c - p_ij
        theta_gradient[i] += diff
        beta_gradient[j] -= diff

    # Update gradient parameters
    theta += lr * theta_gradient
    beta += lr * beta_gradient
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta


def irt(data, val_data, lr, iterations):
    """Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    # TODO: Initialize theta and beta.
    theta = np.zeros(max(data["user_id"]) + 1)
    beta = np.zeros(max(data["question_id"]) + 1)

    val_acc_lst = []
    train_lld_lst = []
    val_lld_lst = []

    for i in range(iterations):
        neg_lld = neg_log_likelihood(data, theta, beta)
        val_neg_lld = neg_log_likelihood(val_data, theta, beta)
        train_lld_lst.append(-neg_lld)
        val_lld_lst.append(-val_neg_lld)
        val_score = evaluate(val_data, theta, beta)
        val_acc_lst.append(val_score)
        print(f"Iteration {i + 1}: NLLK = {neg_lld:.4f}, Val Score = {val_score:.4f}")

        # Update parameters
        theta, beta = update_theta_beta(data, lr, theta, beta)

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, val_acc_lst, train_lld_lst, val_lld_lst


def evaluate(data, theta, beta):
    """Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """


    # predictions here in pred
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) / len(data["is_correct"])


# hannah function, i don't wanna mess up ur workflow
def evaluate_with_predictions(data, theta, beta):
    """
    Evaluate the model given data and return the predictions and accuracy.

    :param data: A dictionary {user_id: list, question_id: list, is_correct: list}
    :param theta: Vector of user abilities.
    :param beta: Vector of question difficulties.
    :return: Tuple (predictions, accuracy)
    """
    predictions = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = theta[u] - beta[q]
        p_a = sigmoid(x)
        predictions.append(p_a)

    # Convert predictions to binary labels for accuracy calculation
    binary_preds = np.array(predictions) >= 0.5
    accuracy = np.sum(data["is_correct"] == binary_preds) / len(data["is_correct"])

    return predictions, accuracy

def main():
    train_data = load_train_csv("./data")
    # You may optionally use the sparse matrix.
    # sparse_matrix = load_train_sparse("./data")
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")

    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    # Hyperparameters
    lr = 0.01  # Learning rate
    iterations = 100  # Number of iterations

    # Train the model using IRT
    theta, beta, val_acc_lst, train_lld_lst, val_lld_lst = irt(
        train_data, val_data, lr, iterations
    )

    # Evaluate on validation and test data
    val_acc = evaluate(val_data, theta, beta)
    test_acc = evaluate(test_data, theta, beta)
    print(f"Final Validation Accuracy: {val_acc:.4f}")
    print(f"Final Test Accuracy: {test_acc:.4f}")

    # Plot the training and validation log-likelihoods
    iterations_range = range(1, iterations + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(iterations_range, train_lld_lst, label="Training Log-Likelihood")
    plt.plot(iterations_range, val_lld_lst, label="Validation Log-Likelihood")
    plt.xlabel("Iteration")
    plt.ylabel("Log-Likelihood")
    plt.title("Training and Validation Log-Likelihoods Over Iterations")
    plt.legend()
    plt.grid(True)
    plt.show()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # Implement part (d)                                                #
    #####################################################################
    # Select three questions
    selected_questions = [17, 40, 85]  # Example question indices

    # Create a range of theta values
    theta_values = np.linspace(-5, 5, 100)

    plt.figure(figsize=(10, 6))
    for q in selected_questions:

        if q < len(beta):
            # Compute probability for each theta
            p_theta = sigmoid(theta_values - beta[q])
            plt.plot(theta_values, p_theta, label=f"Question {q}")
        else:
            print(f"Question index {q} is out of bounds.")

    plt.xlabel("Theta (Student Ability)")
    plt.ylabel("Probability of Correct Response")
    plt.title("Probability of Correct Response vs. Student Ability")
    plt.legend()
    plt.grid(True)
    plt.show()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
