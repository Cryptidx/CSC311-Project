import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torch

from utils import (
    load_valid_csv,
    load_public_test_csv,
    load_train_sparse,
)

# TODO: Add documentation
def load_data(base_path="./data"):
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
    # Create a matrix where each row represents a question and each column
    # represents a subject
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


class AutoEncoderWithEmbeddings(nn.Module):
    def __init__(self, num_users, num_questions, num_subjects, k=100):
        super(AutoEncoderWithEmbeddings, self).__init__()

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


def train(model, train_loader, valid_data, zero_train_data, question_subjects, num_epochs, lr, lamb):
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

            # Get subject info for question batch
            question_subject_batch = question_subjects[question_ids]

            optimizer.zero_grad()
            outputs = model(user_ids, question_ids, question_subject_batch)
            loss = criterion(outputs, is_correct)

            # L2 regularization
            l2_reg = 0.0
            for param in model.parameters():
                l2_reg += torch.norm(param)
            loss += lamb * l2_reg

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        train_losses.append(total_loss)

        # Evaluate on validation data
        valid_acc = evaluate(model, zero_train_data, valid_data, question_subjects)
        valid_accuracies.append(valid_acc)

        print(f"Epoch: {epoch + 1}, Loss: {total_loss:.4f}, Validation Accuracy: {valid_acc:.4f}")

        # Early stopping?? # TODO: experiment without early stoppage
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


def evaluate(model, train_data, data, question_subjects):
    # Todo: Fix this function
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

    # compute accuracy
    accuracy = correct / total
    return accuracy


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


def main():
    zero_train_matrix, train_matrix, valid_data, test_data = load_data()
    question_meta = load_question_metadata()
    subject_meta = load_subject_metadata()

    num_users = zero_train_matrix.shape[0]
    num_questions = zero_train_matrix.shape[1]
    num_subjects = subject_meta['subject_id'].max() + 1

    # Question embeddings
    question_subjects = prepare_question_embeddings(question_meta, num_subjects)
    question_subjects = question_subjects  # Already a FloatTensor??

    # Create training dataset and dataloader
    train_dataset = QuestionDataset(train_matrix.numpy())
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Initialize the model
    k = 50  # Latent dimension
    model = AutoEncoderWithEmbeddings(num_users, num_questions, num_subjects, k)

    # Set training parameters
    num_epochs = 50
    lr = 0.001
    lamb = 0.001

    # Train the model
    train_losses, valid_accuracies = train(
        model,
        train_loader,
        valid_data,
        zero_train_matrix,
        question_subjects,
        num_epochs,
        lr,
        lamb,
    )

    # Evaluate on test data
    test_acc = evaluate(model, zero_train_matrix, test_data, question_subjects)
    print(f"Final Test Accuracy: {test_acc:.4f}")

    # Plot the results
    # Todo: implement better plots
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_losses) + 1), train_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Training Loss vs. Epochs")

    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(valid_accuracies) + 1), valid_accuracies)
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.title("Validation Accuracy vs. Epochs")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
