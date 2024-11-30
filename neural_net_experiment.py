import numpy as np
from matplotlib import pyplot as plt
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader

from utils import (
    load_valid_csv,
    load_public_test_csv,
    load_train_sparse,
)


def load_data(base_path="./data"):
    """Load the data in PyTorch Tensor.

    :return: (zero_train_matrix, train_data, valid_data, test_data)
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


def load_question_metadata(base_path="./data"):
    """Load question metadata."""
    question_meta = pd.read_csv(f"{base_path}/question_meta.csv")
    return question_meta


def load_subject_metadata(base_path="./data"):
    """Load subject metadata."""
    subject_meta = pd.read_csv(f"{base_path}/subject_meta.csv")
    return subject_meta


def prepare_question_embeddings(question_meta, num_subjects):
    """Prepare question embeddings based on subject tags."""
    num_questions = question_meta['question_id'].max() + 1
    question_subject_matrix = np.zeros((num_questions, num_subjects))

    for _, row in question_meta.iterrows():
        question_id = int(row['question_id'])
        subject_ids = eval(row['subject_id'])
        for subject_id in subject_ids:
            question_subject_matrix[question_id, subject_id] = 1

    # Convert to torch tensor
    question_subject_tensor = torch.FloatTensor(question_subject_matrix)
    return question_subject_tensor


class BaseAutoEncoder(nn.Module):
    """Baseline AutoEncoder without enhancements."""
    def __init__(self, num_question, k=100):
        super(BaseAutoEncoder, self).__init__()

        # Define linear functions.
        self.g = nn.Linear(num_question, k)
        self.h = nn.Linear(k, num_question)

    def get_weight_norm(self):
        """Return ||W^1||^2 + ||W^2||^2.

        :return: float
        """
        g_w_norm = torch.norm(self.g.weight, 2) ** 2
        h_w_norm = torch.norm(self.h.weight, 2) ** 2
        return g_w_norm + h_w_norm

    def forward(self, inputs):
        code = self.g(inputs)
        code_activated = torch.sigmoid(code)
        pre_activation_out = self.h(code_activated)
        out = torch.sigmoid(pre_activation_out)
        return out


class ImprovedAutoEncoder(nn.Module):
    """Improved AutoEncoder with Embeddings and Metadata."""
    def __init__(self, num_users, num_questions, num_subjects, k=100):
        super(ImprovedAutoEncoder, self).__init__()

        # Embedding layers for users and questions
        self.user_embedding = nn.Embedding(num_users, k)
        self.question_embedding = nn.Embedding(num_questions, k)

        # Linear layers
        self.fc1 = nn.Linear(k * 2 + num_subjects, k)
        self.fc2 = nn.Linear(k, 1)

        # Dropout layer
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, user_ids, question_ids, question_subjects):
        # Get embeddings
        user_embeds = self.user_embedding(user_ids)
        question_embeds = self.question_embedding(question_ids)

        # Concatenate embeddings and subject information
        x = torch.cat([user_embeds, question_embeds, question_subjects], dim=1)

        # Pass through the network
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        out = torch.sigmoid(self.fc2(x))

        return out.squeeze()


class QuestionDataset(Dataset):
    def __init__(self, data):
        self.user_ids = []
        self.question_ids = []
        self.is_correct = []

        for user_id in range(data.shape[0]):
            for question_id in range(data.shape[1]):
                if not np.isnan(data[user_id, question_id]):
                    self.user_ids.append(user_id)
                    self.question_ids.append(question_id)
                    self.is_correct.append(data[user_id, question_id])

    def __len__(self):
        return len(self.is_correct)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.user_ids[idx]),
            torch.tensor(self.question_ids[idx]),
            torch.tensor(self.is_correct[idx]),
        )


def train_base(model, lr, lamb, train_data, zero_train_data, valid_data, num_epoch):
    """Train the baseline neural network."""
    # Set model to training mode
    model.train()

    # Define optimizer and loss function
    optimizer = optim.SGD(model.parameters(), lr=lr)
    num_student = train_data.shape[0]

    train_losses = []
    valid_accuracies = []

    for epoch in range(num_epoch):
        train_loss = 0.0

        for user_id in range(num_student):
            inputs = Variable(zero_train_data[user_id]).unsqueeze(0)
            target = inputs.clone()

            optimizer.zero_grad()
            output = model(inputs)

            # Mask the target to only compute the gradient of valid entries.
            nan_mask = np.isnan(train_data[user_id].unsqueeze(0).numpy())
            target[nan_mask] = output[nan_mask]

            # Compute loss
            error = torch.sum((output - target) ** 2.0)
            weight_norm = model.get_weight_norm()
            overall_weight_norm = (lamb / 2) * weight_norm
            loss = error + overall_weight_norm

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_losses.append(train_loss)

        valid_acc = evaluate_base(model, zero_train_data, valid_data)
        valid_accuracies.append(valid_acc)
        print(
            "Epoch: {} \tTraining Loss: {:.6f}\t Validation Accuracy: {:.4f}".format(
                epoch + 1, train_loss, valid_acc
            )
        )

    return train_losses, valid_accuracies


def train_improved(model, train_loader, valid_data, zero_train_data, question_subjects, num_epochs, lr, lamb):
    """Train the improved neural network with embeddings and metadata."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    best_valid_acc = 0.0
    patience = 5
    patience_counter = 0

    train_losses = []
    valid_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            user_ids, question_ids, is_correct = batch
            user_ids = user_ids.long()
            question_ids = question_ids.long()
            is_correct = is_correct.float()

            # Get subject information for the batch of questions
            question_subject_batch = question_subjects[question_ids]

            optimizer.zero_grad()
            outputs = model(user_ids, question_ids, question_subject_batch)
            loss = criterion(outputs, is_correct)

            # L2 regularization
            l2_reg = 0.0
            for param in model.parameters():
                l2_reg += torch.norm(param)
            loss += (lamb / 2) * l2_reg

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        train_losses.append(total_loss)

        # Evaluate on validation data
        valid_acc = evaluate_improved(model, zero_train_data, valid_data, question_subjects)
        valid_accuracies.append(valid_acc)

        print(f"Epoch: {epoch + 1}, Loss: {total_loss:.4f}, Validation Accuracy: {valid_acc:.4f}")

        # Early stopping
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pth")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    # Load the best model
    model.load_state_dict(torch.load("best_model.pth"))

    return train_losses, valid_accuracies


def evaluate_base(model, train_data, data):
    """Evaluate the baseline model."""
    # Set model to evaluation mode
    model.eval()

    total = 0
    correct = 0

    for i in range(len(data["user_id"])):
        user_id = data["user_id"][i]
        question_id = data["question_id"][i]
        is_correct = data["is_correct"][i]

        inputs = Variable(train_data[user_id]).unsqueeze(0)
        output = model(inputs)

        pred = output[0, question_id] >= 0.5

        if pred.item() == is_correct:
            correct += 1
        total += 1

    accuracy = correct / total
    return accuracy


def evaluate_improved(model, train_data, data, question_subjects):
    """Evaluate the improved model."""
    model.eval()
    total = 0
    correct = 0

    user_ids = torch.tensor(data["user_id"]).long()
    question_ids = torch.tensor(data["question_id"]).long()
    is_correct = torch.tensor(data["is_correct"]).float()

    # Get subject information for the questions
    question_subject_batch = question_subjects[question_ids]

    with torch.no_grad():
        outputs = model(user_ids, question_ids, question_subject_batch)
        preds = outputs >= 0.5
        correct += torch.sum(preds == is_correct.byte()).item()
        total += len(is_correct)

    accuracy = correct / total
    return accuracy


def main():
    zero_train_matrix, train_matrix, valid_data, test_data = load_data()
    question_meta = load_question_metadata()
    subject_meta = load_subject_metadata()

    num_users = zero_train_matrix.shape[0]
    num_questions = zero_train_matrix.shape[1]
    num_subjects = subject_meta['subject_id'].max() + 1

    # Prepare question subject embeddings
    question_subjects = prepare_question_embeddings(question_meta, num_subjects)
    question_subjects = question_subjects  # Already a FloatTensor

    # Create training dataset and dataloader for improved model
    train_dataset = QuestionDataset(train_matrix.numpy())
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Initialize models
    k = 100  # Latent dimension (set to match your original code)
    base_model = BaseAutoEncoder(num_questions, k)
    improved_model = ImprovedAutoEncoder(num_users, num_questions, num_subjects, k)

    # Set training parameters
    num_epochs = 65  # Match your original number of epochs
    lr_base = 0.005  # Learning rate for baseline model
    lr_improved = 0.005  # Learning rate for improved model
    lamb = 0.001  # Regularization parameter

    # Train the baseline model
    print("Training the baseline model...")
    base_train_losses, base_valid_accuracies = train_base(
        base_model, lr_base, lamb, train_matrix, zero_train_matrix, valid_data, num_epochs
    )

    # Evaluate the baseline model
    base_test_acc = evaluate_base(base_model, zero_train_matrix, test_data)
    print(f"Baseline Model Test Accuracy: {base_test_acc:.4f}")

    # Train the improved model
    print("\nTraining the improved model...")
    improved_train_losses, improved_valid_accuracies = train_improved(
        improved_model,
        train_loader,
        valid_data,
        zero_train_matrix,
        question_subjects,
        num_epochs,
        lr_improved,
        lamb,
    )

    # Evaluate the improved model
    improved_test_acc = evaluate_improved(improved_model, zero_train_matrix, test_data, question_subjects)
    print(f"Improved Model Test Accuracy: {improved_test_acc:.4f}")

    # Plot the results
    epochs_base = range(1, len(base_train_losses) + 1)
    epochs_improved = range(1, len(improved_train_losses) + 1)

    plt.figure(figsize=(12, 5))

    # Training Loss Comparison
    plt.subplot(1, 2, 1)
    plt.plot(epochs_base, base_train_losses, label='Baseline Model')
    plt.plot(epochs_improved, improved_train_losses, label='Improved Model')
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Training Loss vs. Epochs")
    plt.legend()

    # Validation Accuracy Comparison
    plt.subplot(1, 2, 2)
    plt.plot(epochs_base, base_valid_accuracies, label='Baseline Model')
    plt.plot(epochs_improved, improved_valid_accuracies, label='Improved Model')
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.title("Validation Accuracy vs. Epochs")
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Create a table for comparison
    print("\nComparison of Models:")
    print(f"{'Model':<20} {'Validation Accuracy':<20} {'Test Accuracy':<15}")
    print(f"{'-'*55}")
    print(f"{'Baseline Model':<20} {base_valid_accuracies[-1]:<20.4f} {base_test_acc:<15.4f}")
    print(f"{'Improved Model':<20} {improved_valid_accuracies[-1]:<20.4f} {improved_test_acc:<15.4f}")


if __name__ == "__main__":
    main()

