import torch
import torch.nn as nn

class LSTMQuestionEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=1024):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)

    def forward(self, question_tokens):
        x = self.embedding(question_tokens)
        _, (h_n, _) = self.lstm(x)
        return h_n.squeeze(0) 
