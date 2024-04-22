from datasets import load_dataset
import gensim.downloader as api
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch.nn.utils.rnn as rnn_utils
import senteval
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train models for SNLI task")
    parser.add_argument("model_type", type=str, choices=["baseline", "udlstm", "bilstm", "bilstm-max"], help="Type of model to train")
    parser.add_argument("checkpoint_path", type=str, nargs='?', help="Path to save the best model checkpoint")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--lr_threshold", type=float, default=10e-5, help="Learning rate threshold for early stopping")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    return parser.parse_args()

args = parse_arguments()
initial_lr = args.lr
lr_threshold = args.lr_threshold
num_epochs = args.epochs

# Check if GPU is available, otherwise use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load datasets
train_dataset = load_dataset('stanfordnlp/snli', split='train')
val_dataset = load_dataset('stanfordnlp/snli', split='validation')
test_dataset = load_dataset('stanfordnlp/snli', split='test')

# download Glove embeddings
Glove_model = api.load("glove-wiki-gigaword-300")

# add unseen word token
UNK_TOKEN = "<UNK>"
vocabulary = {UNK_TOKEN: np.random.rand(Glove_model.vector_size)}

def vocab(datapoint):
    tokenised = datapoint.lower().split()
    for token in tokenised:
        if token in Glove_model:
            vocabulary[token] = Glove_model[token]
        else:
            vocabulary[token] = vocabulary[UNK_TOKEN]
    return vocab

for dataset in [train_dataset]:
    for datapoint in dataset:
        vocab(datapoint['premise'])
        vocab(datapoint['hypothesis'])

# check longest datapoint for padding
longest = 0
for dataset in [train_dataset]:
    for datapoint in dataset:
        length = len(datapoint['premise'].lower().split())
        if length > longest:
            longest = length

        length = len(datapoint['hypothesis'].lower().split())
        if length > longest:
            longest = length
print(longest)

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fully_connected_1 = nn.Linear(input_size, hidden_size)
        self.fully_connected_2 = nn.Linear(hidden_size, hidden_size)
        self.fully_connected_3 = nn.Linear(hidden_size, output_size)
        
        self.seq = nn.Sequential(
            self.fully_connected_1,
            self.fully_connected_2,
            self.fully_connected_3,
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.seq(x)


# Set up TensorBoard writer
writer = SummaryWriter()

output_size = 3
best_val_loss = float('inf')
best_model_state_dict = None
best_epoch = -1

def train_model(model, train_loader, val_loader, loss_function, optimizer, writer, num_epochs):
    global initial_lr
    global lr_threshold
    global best_val_loss
    global best_model_state_dict
    global best_epoch

    for epoch in range(num_epochs):
        # first train model
        model.train()
        all_loss = 0
        for premise_hypothesis, label in train_loader:

            premise_hypothesis = premise_hypothesis.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            output = model(premise_hypothesis)
            loss = loss_function(output, label)
            loss.backward()
            optimizer.step()
            all_loss += loss
        trainloss = all_loss / len(train_loader)

        # evaluate model
        model.eval()
        all_val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for premise_hypothesis, label in val_loader:

                premise_hypothesis = premise_hypothesis.to(device)
                label = label.to(device)

                output = model(premise_hypothesis)
                loss = loss_function(output, label)
                all_val_loss += loss

                _, predicted = torch.max(output, 1)
                total += label.size(0)
                correct += (predicted == label).sum()
        valloss = all_val_loss / len(val_loader)
        val_accuracy = correct / total

        # Check if current model is the best so far
        if valloss < best_val_loss:
            best_val_loss = valloss
            best_model_state_dict = model.state_dict()
            best_epoch = epoch
            if args.checkpoint_path:
                save_path = args.checkpoint_path
                print("checkpoint saved to path")
            elif args.model_type == "baseline":
                save_path = 'best_model_checkpoint_baseline1.pth'
                print("checkpoint saved to best_model_checkpoint_baseline1.pth")
            elif args.model_type == "udlstm":
                save_path = 'best_model_checkpoint_UDLSTM1.pth'
                print("checkpoint saved to best_model_checkpoint_UDLSTM1.pth")
            elif args.model_type == "bilstm":
                save_path = 'best_model_checkpoint_BiLSTM1.pth'
                print("checkpoint saved to best_model_checkpoint_BiLSTM1.pth")
            elif args.model_type == "bilstm-max":
                save_path = 'best_model_checkpoint_BiLSTM_MAX1.pth'
                print("checkpoint saved to best_model_checkpoint_BiLSTM_MAX1.pth")
            else:
                save_path = 'best_model_checkpoint.pth'
                print("checkpoint saved to best_model_checkpoint.pth")
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_model_state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss
            }, save_path)

        else:
            # If validation loss increases, reduce learning rate
            initial_lr = initial_lr / 5
            for param_group in optimizer.param_groups:
                param_group['lr'] = initial_lr

        # early stopping
        if initial_lr < lr_threshold:
            print("Learning rate below threshold so we stop training.")
            break

        writer.add_scalar('Loss/train', trainloss, epoch)
        writer.add_scalar('Loss/val', valloss, epoch)
        writer.add_scalar('Accuracy/val', val_accuracy, epoch)
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {trainloss:.4f}, Val Loss: {valloss:.4f}, Val Acc: {val_accuracy:.2%}')
        writer.close()





if args.model_type == "baseline":
    def get_embeddings_baseline(datapoint):
        tokenised = datapoint.lower().split()
        # already takes care of padding by taking out all embeddings where all is 0
        embeddings = [vocabulary[token] for token in tokenised if token in vocabulary and not np.all(vocabulary[token] == 0)]
        if len(embeddings) == 0:
            return np.zeros(Glove_model.vector_size)
        return np.mean(embeddings, axis=0)

    def Baseline_dataset(data):
        all_embeddings_baseline = {}
        for datapoint in data:
            premise_embedding = get_embeddings_baseline(datapoint['premise'])
            hypothesis_embedding = get_embeddings_baseline(datapoint['hypothesis'])

            # Check if premise_embedding and hypothesis_embedding have the same number of dimensions
            if premise_embedding.shape[0] != hypothesis_embedding.shape[0]:
                print(f"Skipping datapoint: {datapoint}, premise_embedding and hypothesis_embedding have different dimensions")
                continue

            # Convert numpy arrays to PyTorch tensors
            concat_embeddings = torch.tensor(np.concatenate([premise_embedding, hypothesis_embedding]), dtype=torch.float32)
            elementwise_embeddings = torch.tensor(premise_embedding * hypothesis_embedding, dtype=torch.float32)
            abs_diff_embeddings = torch.tensor(np.abs(premise_embedding - hypothesis_embedding), dtype=torch.float32)

            embeddings = torch.cat([concat_embeddings, elementwise_embeddings, abs_diff_embeddings], dim=0)
            # all_embeddings_baseline[torch.tensor(embeddings, dtype=torch.float32)] = torch.tensor(datapoint['label'], dtype=torch.long)
            all_embeddings_baseline[torch.tensor(embeddings, dtype=torch.float32).clone().detach()] = torch.tensor(datapoint['label'], dtype=torch.long).clone().detach()
        return all_embeddings_baseline

    # baseline version
    baseline_train_data = Baseline_dataset(train_dataset)
    baseline_val_data = Baseline_dataset(val_dataset)
    baseline_test_data = Baseline_dataset(test_dataset)

    input_size = 1200
    hidden_size = 512
    output_size = 3
    MLP_model = MLP(input_size, hidden_size, output_size)
    MLP_model = MLP_model.to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(MLP_model.parameters(), lr=0.1, weight_decay=0.99)

    # Convert baseline_train_data dictionary to a list of tuples and filter -1 labels
    train_data_list = [(embedding, label) for embedding, label in baseline_train_data.items() if label != -1]
    val_data_list = [(embedding, label) for embedding, label in baseline_val_data.items() if label != -1]
    test_data_list = [(embedding, label) for embedding, label in baseline_test_data.items() if label != -1]

    train_loader = DataLoader(train_data_list, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_data_list, batch_size=64)
    test_loader = DataLoader(test_data_list, batch_size=64)

    train_model(MLP_model, train_loader, val_loader, loss_function, optimizer, writer, num_epochs)


elif args.model_type == "udlstm":
    class LSTMEncoder(nn.Module):
        def __init__(self, embedding_dim, hidden_dim):
            super(LSTMEncoder, self).__init__()
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

        def forward(self, embedded, lengths):
            packed = rnn_utils.pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
            _, (hidden, _) = self.lstm(packed.float())
            return hidden.squeeze(0)

    input_size = 300
    hidden_size = 512
    lstm_encoder = LSTMEncoder(input_size, hidden_size)

    def get_embeddings_UDLSTM(datapoint, encoder):
        lengths_list = []
        tokenised = datapoint.lower().split()
        indexed = [vocabulary[token] for token in tokenised if token in vocabulary and not np.all(vocabulary[token] == 0)]
        if len(indexed) == 0:
            return np.zeros(Glove_model.vector_size)
        lengths_list.append(len(indexed))
        # Add batch dimension (right now batch is 1)
        indexed = torch.tensor(indexed).unsqueeze(0)
        # pad
        indexed = torch.nn.utils.rnn.pad_sequence(indexed, batch_first=True, padding_value=1)
        embedding = encoder(indexed, torch.tensor(lengths_list))
        return embedding.detach().numpy()

    def UDLSTM_dataset(data, lstm_encoder):
        all_embeddings = {}
        for datapoint in data:
            premise_embedding = get_embeddings_UDLSTM(datapoint['premise'], lstm_encoder)
            hypothesis_embedding = get_embeddings_UDLSTM(datapoint['hypothesis'], lstm_encoder)

            # Check if premise_embedding and hypothesis_embedding have the same number of dimensions
            if premise_embedding.shape[0] != hypothesis_embedding.shape[0]:
                print(f"Skipping datapoint: {datapoint}, premise_embedding and hypothesis_embedding have different dimensions")
                continue

            concat_embeddings = torch.cat([torch.tensor(premise_embedding, dtype=torch.float32), torch.tensor(hypothesis_embedding, dtype=torch.float32)], dim=1)
            elementwise_embeddings = torch.tensor(premise_embedding * hypothesis_embedding, dtype=torch.float32)
            abs_diff_embeddings = torch.tensor(np.abs(premise_embedding - hypothesis_embedding), dtype=torch.float32)

            embeddings = torch.cat([concat_embeddings, elementwise_embeddings, abs_diff_embeddings], dim=1).squeeze(0)
            all_embeddings[torch.tensor(embeddings, dtype=torch.float32).clone().detach()] = torch.tensor(datapoint['label'], dtype=torch.long).clone().detach()
        return all_embeddings

    # UDLSTM version
    UDLSTM_train_data = UDLSTM_dataset(train_dataset, lstm_encoder)
    UDLSTM_val_data = UDLSTM_dataset(val_dataset, lstm_encoder)
    UDLSTM_test_data = UDLSTM_dataset(test_dataset, lstm_encoder)

    input_size = 2048
    UDLSTM_MLP_model = MLP(input_size, hidden_size, output_size)
    UDLSTM_MLP_model = UDLSTM_MLP_model.to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(UDLSTM_MLP_model.parameters(), lr=0.1, weight_decay=0.99)

    # Convert baseline_train_data dictionary to a list of tuples and filter -1 labels
    UDLSTM_train_data_list = [(embedding, label) for embedding, label in UDLSTM_train_data.items() if label != -1]
    UDLSTM_val_data_list = [(embedding, label) for embedding, label in UDLSTM_val_data.items() if label != -1]
    UDLSTM_test_data_list = [(embedding, label) for embedding, label in UDLSTM_test_data.items() if label != -1]

    UDLSTM_train_loader = DataLoader(UDLSTM_train_data_list, batch_size=64, shuffle=True)
    UDLSTM_val_loader = DataLoader(UDLSTM_val_data_list, batch_size=64)
    UDLSTM_test_loader = DataLoader(UDLSTM_test_data_list, batch_size=64)

    train_model(UDLSTM_MLP_model, UDLSTM_train_loader, UDLSTM_val_loader, loss_function, optimizer, writer, num_epochs=5)


elif args.model_type == "bilstm":
    class BiLSTMEncoder(nn.Module):
        def __init__(self, embedding_dim, hidden_dim):
            super(BiLSTMEncoder, self).__init__()
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)

        def forward(self, embedded, lengths):
            packed = rnn_utils.pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
            _, (hidden, _) = self.lstm(packed.float())
            # Concatenate the last hidden states of forward and backward LSTMs
            concatenated = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
            return concatenated

    input_size = 300
    hidden_size = 512
    Bilstm_encoder = BiLSTMEncoder(input_size, hidden_size)
    
    def get_embeddings_BiLSTM(datapoint, encoder):
        lengths_list = []
        tokenised = datapoint.lower().split()
        indexed = [vocabulary[token] for token in tokenised if token in vocabulary and not np.all(vocabulary[token] == 0)]
        if len(indexed) == 0:
            return np.zeros(Glove_model.vector_size)
        lengths_list.append(len(indexed))
        indexed = torch.tensor(indexed).unsqueeze(0)
        indexed = torch.nn.utils.rnn.pad_sequence(indexed, batch_first=True, padding_value=1)
        embedding = encoder(indexed, torch.tensor(lengths_list))
        return embedding.detach().numpy()

    def BiLSTM_dataset(data, lstm_encoder):
        all_embeddings = {}
        for datapoint in data:
            premise_embedding = get_embeddings_BiLSTM(datapoint['premise'], lstm_encoder)
            hypothesis_embedding = get_embeddings_BiLSTM(datapoint['hypothesis'], lstm_encoder)

            # Check if premise_embedding and hypothesis_embedding have the same number of dimensions
            if premise_embedding.shape[0] != hypothesis_embedding.shape[0]:
                print(f"Skipping datapoint: {datapoint}, premise_embedding and hypothesis_embedding have different dimensions")
                continue

            concat_embeddings = torch.cat([torch.tensor(premise_embedding, dtype=torch.float32), torch.tensor(hypothesis_embedding, dtype=torch.float32)], dim=1)
            elementwise_embeddings = torch.tensor(premise_embedding * hypothesis_embedding, dtype=torch.float32)
            abs_diff_embeddings = torch.tensor(np.abs(premise_embedding - hypothesis_embedding), dtype=torch.float32)

            embeddings = torch.cat([concat_embeddings, elementwise_embeddings, abs_diff_embeddings], dim=1).squeeze(0)
            all_embeddings[torch.tensor(embeddings, dtype=torch.float32).clone().detach()] = torch.tensor(datapoint['label'], dtype=torch.long).clone().detach()
        return all_embeddings

    # BiLSTM version
    BiLSTM_train_data = BiLSTM_dataset(train_dataset, Bilstm_encoder)
    BiLSTM_val_data = BiLSTM_dataset(val_dataset, Bilstm_encoder)
    BiLSTM_test_data = BiLSTM_dataset(test_dataset, Bilstm_encoder)

    input_size = 4096
    BiLSTM_MLP_model = MLP(input_size, hidden_size, output_size)
    BiLSTM_MLP_model = BiLSTM_MLP_model.to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(BiLSTM_MLP_model.parameters(), lr=0.1, weight_decay=0.99)

    # Convert baseline_train_data dictionary to a list of tuples and filter -1 labels
    BiLSTM_train_data_list = [(embedding, label) for embedding, label in BiLSTM_train_data.items() if label != -1]
    BiLSTM_val_data_list = [(embedding, label) for embedding, label in BiLSTM_val_data.items() if label != -1]
    BiLSTM_test_data_list = [(embedding, label) for embedding, label in BiLSTM_test_data.items() if label != -1]

    BiLSTM_train_loader = DataLoader(BiLSTM_train_data_list, batch_size=64, shuffle=True)
    BiLSTM_val_loader = DataLoader(BiLSTM_val_data_list, batch_size=64)
    BiLSTM_test_loader = DataLoader(BiLSTM_test_data_list, batch_size=64)

    train_model(BiLSTM_MLP_model, BiLSTM_train_loader, BiLSTM_val_loader, loss_function, optimizer, writer, num_epochs=5)


elif args.model_type == "bilstm-max":
    class BiLSTMMaxEncoder(nn.Module):
        def __init__(self, embedding_dim, hidden_dim):
            super(BiLSTMMaxEncoder, self).__init__()
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)

        def forward(self, embedded, lengths):
            packed = rnn_utils.pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
            _, (hidden, _) = self.lstm(packed.float())
            pooled, _ = torch.max(hidden, dim=0)
            
            return pooled

    input_size = 300
    hidden_size = 512
    Bilstm_MAX_encoder = BiLSTMMaxEncoder(input_size, hidden_size)
    Bilstm_MAX_encoder = Bilstm_MAX_encoder.to(device)
    
    def get_embeddings_BiLSTM_MAX(datapoint, encoder):
        lengths_list = []
        tokenised = datapoint.lower().split()
        indexed = [vocabulary[token] for token in tokenised if token in vocabulary and not np.all(vocabulary[token] == 0)]
        if len(indexed) == 0:
            return np.zeros(Glove_model.vector_size)
        lengths_list.append(len(indexed))
        indexed = torch.tensor(indexed).unsqueeze(0)
        indexed = torch.nn.utils.rnn.pad_sequence(indexed, batch_first=True, padding_value=1)
        embedding = encoder(indexed, torch.tensor(lengths_list))
        return embedding.detach().numpy()


    def BiLSTM_MAX_dataset(data, lstm_encoder):
        all_embeddings = {}
        for datapoint in data:
            premise_embedding = get_embeddings_BiLSTM_MAX(datapoint['premise'], lstm_encoder)
            hypothesis_embedding = get_embeddings_BiLSTM_MAX(datapoint['hypothesis'], lstm_encoder)

            # Check if premise_embedding and hypothesis_embedding have the same number of dimensions
            if premise_embedding.shape[0] != hypothesis_embedding.shape[0]:
                print(f"Skipping datapoint: {datapoint}, premise_embedding and hypothesis_embedding have different dimensions")
                continue

            concat_embeddings = torch.cat([torch.tensor(premise_embedding, dtype=torch.float32), torch.tensor(hypothesis_embedding, dtype=torch.float32)], dim=1)
            elementwise_embeddings = torch.tensor(premise_embedding * hypothesis_embedding, dtype=torch.float32)
            abs_diff_embeddings = torch.tensor(np.abs(premise_embedding - hypothesis_embedding), dtype=torch.float32)

            embeddings = torch.cat([concat_embeddings, elementwise_embeddings, abs_diff_embeddings], dim=1).squeeze(0)
            all_embeddings[torch.tensor(embeddings, dtype=torch.float32).clone().detach()] = torch.tensor(datapoint['label'], dtype=torch.long).clone().detach()
        return all_embeddings

    # BiLSTM_MAX version
    BiLSTM_MAX_train_data = BiLSTM_MAX_dataset(train_dataset, Bilstm_MAX_encoder)
    BiLSTM_MAX_val_data = BiLSTM_MAX_dataset(val_dataset, Bilstm_MAX_encoder)
    BiLSTM_MAX_test_data = BiLSTM_MAX_dataset(test_dataset, Bilstm_MAX_encoder)

    input_size = 2048
    BiLSTM_MAX_MLP_model = MLP(input_size, hidden_size, output_size)
    BiLSTM_MAX_MLP_model = BiLSTM_MAX_MLP_model.to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(BiLSTM_MAX_MLP_model.parameters(), lr=0.1, weight_decay=0.99)

    # Convert baseline_train_data dictionary to a list of tuples and filter -1 labels
    BiLSTM_MAX_train_data_list = [(embedding, label) for embedding, label in BiLSTM_MAX_train_data.items() if label != -1]
    BiLSTM_MAX_val_data_list = [(embedding, label) for embedding, label in BiLSTM_MAX_val_data.items() if label != -1]
    BiLSTM_MAX_test_data_list = [(embedding, label) for embedding, label in BiLSTM_MAX_test_data.items() if label != -1]

    BiLSTM_MAX_train_loader = DataLoader(BiLSTM_MAX_train_data_list, batch_size=64, shuffle=True)
    BiLSTM_MAX_val_loader = DataLoader(BiLSTM_MAX_val_data_list, batch_size=64)
    BiLSTM_MAX_test_loader = DataLoader(BiLSTM_MAX_test_data_list, batch_size=64)

    train_model(BiLSTM_MAX_MLP_model, BiLSTM_MAX_train_loader, BiLSTM_MAX_val_loader, loss_function, optimizer, writer, num_epochs=5)
