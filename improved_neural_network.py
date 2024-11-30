import numpy as np
from matplotlib import pyplot as plt
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torch


def load_data(base_path="/content/drive/MyDrive/data"):
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
    train_matrix = load_train_sparse(base_path).toarray()
    valid_data = load_valid_csv(base_path)
    test_data = load_public_test_csv(base_path)

    zero_train_matrix = train_matrix.copy()
    # Fill in the missing entries to 0.
    zero_train_matrix[np.isnan(train_matrix)] = 0
    # Change to Float Tensor for PyTorch.
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    train_matrix = torch.FloatTensor(train_matrix)

    return zero_train_matrix, train_matrix, valid_data, test_data


class AutoEncoder(nn.Module):
    def __init__(self, num_question, k=100):
        """Initialize a class AutoEncoder.

        :param num_question: int
        :param k: int
        """
        super(AutoEncoder, self).__init__()

        # Define the layers
        self.fc1 = nn.Linear(num_question, num_question // 2)
        self.fc2 = nn.Linear(num_question // 2, k)
        self.fc3 = nn.Linear(k, num_question // 2)
        self.fc4 = nn.Linear(num_question // 2, num_question)

    def get_weight_norm(self):
        """Return the L2 norm of the weights."""
        norm = 0.0
        for name, param in self.named_parameters():
            if 'weight' in name:
                norm += torch.norm(param, 2) ** 2
        return norm

    def forward(self, inputs):
        """Forward pass.

        :param inputs: user vector.
        :return: reconstructed user vector.
        """
        # Relu Activation function
        intermediate = self.fc1(inputs)
        intermediate_activated = F.relu(intermediate)
        code = self.fc2(intermediate_activated)
        code_activated = F.relu(code)
        second_intermediate = self.fc3(code_activated)
        second_intermediate_activated = F.relu(second_intermediate)
        out = self.fc4(second_intermediate_activated)
        out_activated = torch.sigmoid(out)

        return out_activated


def train(model, lr, lamb, train_data, zero_train_data, valid_data, num_epoch):
    """Train the neural network with regularization and binary cross-entropy loss.

    :param model: AutoEncoder model
    :param lr: float, learning rate
    :param lamb: float, regularization strength
    :param train_data: 2D FloatTensor, training data with NaN for missing entries
    :param zero_train_data: 2D FloatTensor, training data with 0 for missing entries
    :param valid_data: dict, validation data
    :param num_epoch: int, number of epochs
    :return: list of training losses, list of validation accuracies
    """
    # Set model to training mode
    model.train()

    # Define optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    num_student = train_data.shape[0]

    train_losses = []
    valid_accuracies = []

    for epoch in range(num_epoch):
        train_loss = 0.0

        for user_id in range(num_student):
            inputs = Variable(zero_train_data[user_id]).unsqueeze(0)
            target = train_data[user_id].unsqueeze(0)

            optimizer.zero_grad()
            output = model(inputs)

            # Mask to zero out the missing entries in target
            nan_mask = torch.isnan(target)
            masked_target = torch.where(nan_mask, torch.tensor(0.0, device=target.device), target)
            masked_output = torch.where(nan_mask, torch.tensor(0.0, device=output.device), output)

            # Compute binary cross-entropy loss
            loss = criterion(masked_output, masked_target)

            # Add L2 regularization
            loss += (lamb / 2) * model.get_weight_norm()

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_losses.append(train_loss)

        valid_acc = evaluate(model, zero_train_data, valid_data)
        valid_accuracies.append(valid_acc)
        print(
            "Epoch: {} \tTraining Loss: {:.6f}\t Validation Accuracy: {:.4f}".format(
                epoch + 1, train_loss, valid_acc
            )
        )

    return train_losses, valid_accuracies


def evaluate(model, train_data, valid_data):
    """Evaluate the model on validation or test data.

    :param model: AutoEncoder model
    :param train_data: 2D FloatTensor
    :param valid_data: dict, validation or test data
    :return: float, accuracy
    """
    # Set model to evaluation mode
    model.eval()

    total = 0
    correct = 0

    for i in range(len(valid_data["user_id"])):
        user_id = valid_data["user_id"][i]
        question_id = valid_data["question_id"][i]
        is_correct = valid_data["is_correct"][i]

        inputs = Variable(train_data[user_id]).unsqueeze(0)
        output = model(inputs)

        pred = output[0, question_id] >= 0.5

        if pred.item() == is_correct:
            correct += 1
        total += 1

    accuracy = correct / total
    return accuracy


def main():
    zero_train_matrix, train_matrix, valid_data, test_data = load_data()

    #####################################################################
    # TODO:                                                             #
    # Try out 5 different k and select the best k using the             #
    # validation set.                                                   #
    #####################################################################
    # Set model hyperparameters
    k = 100  # Latent dimension
    num_questions = zero_train_matrix.shape[1]
    model = AutoEncoder(num_questions, k)

    # Set optimization hyperparameters
    lr = 0.01  # Learning rate
    num_epoch = 65
    lamb = 0.001  # Regularization strength

    train_losses, valid_accuracies = train(
        model, lr, lamb, train_matrix, zero_train_matrix, valid_data, num_epoch
    )

    # Evaluate on validation data
    final_valid_acc = evaluate(model, zero_train_matrix, valid_data)
    print(f"Final Validation Accuracy: {final_valid_acc:.4f}")

    # Evaluate on test data
    final_test_acc = evaluate(model, zero_train_matrix, test_data)
    print(f"Final Test Accuracy: {final_test_acc:.4f}")

    # Plot training loss and validation accuracy
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epoch + 1), train_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Training Loss vs. Epochs")

    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epoch + 1), valid_accuracies)
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.title("Validation Accuracy vs. Epochs")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
