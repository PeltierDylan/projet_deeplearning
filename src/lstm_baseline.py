import torch
import torch.nn as nn
import torch.optim as optim
import random
from tqdm import tqdm

# Définition des tokens spéciaux
SOS_token = 0
EOS_token = 1
PAD_token = 2
UNK_token = 3

class Lang:
    """Classe pour construire le vocabulaire à partir du texte brut (From Scratch)"""
    def __init__(self, name):
        self.name = name
        self.word2index = {"<SOS>": SOS_token, "<EOS>": EOS_token, "<PAD>": PAD_token, "<UNK>": UNK_token}
        self.index2word = {SOS_token: "<SOS>", EOS_token: "<EOS>", PAD_token: "<PAD>", UNK_token: "<UNK>"}
        self.n_words = 4  # Compte les mots uniques

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1

class EncoderRNN(nn.Module):
    """L'Encodeur lit la phrase source (Français) et crée un vecteur de contexte."""
    def __init__(self, input_size, hidden_size, dropout_p=0.1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.lstm(embedded)
        return output, hidden

class DecoderRNN(nn.Module):
    """Le Décodeur prend le contexte et génère la phrase cible (Anglais) mot par mot."""
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        output = self.embedding(input)
        output = torch.relu(output)
        output, hidden = self.lstm(output, hidden)
        output = self.out(output)
        return output, hidden

def tensorFromSentence(lang, sentence, device):
    """Convertit une phrase en un tenseur d'indices compréhensible par le modèle."""
    indexes = [lang.word2index.get(word, UNK_token) for word in sentence.split(' ')]
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(1, -1)

def train_epoch(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, device):
    """Effectue un pas d'entraînement sur une phrase (Forward + Backward pass)."""
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    encoder_outputs, encoder_hidden = encoder(input_tensor)

    decoder_input = torch.tensor([[SOS_token]], device=device)
    decoder_hidden = encoder_hidden

    loss = 0
    target_length = target_tensor.size(1)

    # Teacher forcing : on donne le "vrai" mot précédent comme input pour accélérer l'apprentissage
    use_teacher_forcing = True if random.random() < 0.5 else False

    if use_teacher_forcing:
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            loss += criterion(decoder_output.view(1, -1), target_tensor[0, di].view(1))
            decoder_input = target_tensor[0, di].view(1, -1)
    else:
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach().view(1, -1)
            loss += criterion(decoder_output.view(1, -1), target_tensor[0, di].view(1))
            if decoder_input.item() == EOS_token:
                break

    loss.backward() # Calcul des gradients (Backpropagation)
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

def evaluate_lstm(encoder, decoder, sentence, input_lang, output_lang, device, max_length=50):
    """Fonction d'inférence (Traduction d'une phrase)."""
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence, device)
        encoder_outputs, encoder_hidden = encoder(input_tensor)

        decoder_input = torch.tensor([[SOS_token]], device=device)
        decoder_hidden = encoder_hidden
        decoded_words = []

        for di in range(max_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            if topi.item() == EOS_token:
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])
            decoder_input = topi.squeeze().detach().view(1, -1)

        return " ".join(decoded_words)