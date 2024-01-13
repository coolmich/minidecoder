import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
from util import *


dropout_rate = 0.2

class BigramLM(nn.Module):  # loss in the 2.5-2.6 range
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

    # (B, T) -> (B, T, C), C is the embedding dim
    # embedding dim same as vocab size, cuz it acts as one-hot for output
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(input)
        return emb


# input: context, output: weight to combine context embeddings
# lookback window depends on timestep, the model is flexible
"""
# 1. The K, W, Q are NOT lookup embeddings given a vocab index
#    They are linear projections from upstream embeddings
# 2. The in/out is the same during training as Bigram,
#    input: (Ti, Ti+1...Tj), output pred of (Ti+1, Ti+2 ... Tj+1)
# 3. Therefore attention is done at every T,
#    and mask is used to avoid attending to later T
"""
class AttentionHead(nn.Module):
    def __init__(self, input_dim, head_dim):
        super().__init__()
        self.k = nn.Linear(input_dim, head_dim)
        self.q = nn.Linear(input_dim, head_dim)
        self.v = nn.Linear(input_dim, head_dim)
        self.drop = nn.Dropout(dropout_rate)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # (B, T, C) -> (B, T, head_dim)
        K = self.k(input).permute(0, 2, 1)  # (B, head_dim, T)
        Q = self.q(input)
        """
        About Tril
        # Naive avg of previous timesteps
        # tri = torch.tril(torch.ones(3, 3))
        # tri[tri == 0] = float('-inf')
        # F.softmax(tri, dim=1)
        >> tensor([[1.0000, 0.0000, 0.0000],
            [0.5000, 0.5000, 0.0000],
            [0.3333, 0.3333, 0.3333]])

        # Our job is to do better than avg
        # So self attention is to derive something better than torch.ones(3, 3)
        """
        wei = torch.tril(Q @ K) # (B, T, T)
        wei[wei == 0] = float('-inf')
        norm_wei = F.softmax(wei, dim=-1)
        return self.drop(norm_wei) @ self.v(input)


class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, num_heads, head_dim):
        super().__init__()
        self.attention_heads = nn.ModuleList([AttentionHead(input_dim, head_dim) for _ in range(num_heads)])
        self.proj = nn.Linear(head_dim * num_heads, input_dim)
        self.drop = nn.Dropout(dropout_rate)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # (B, T, C) -> (B, T, head_dim * num_heads)
        cat = torch.cat([h(input) for h in self.attention_heads], dim=-1)
        return self.drop(self.proj(cat))



class FeedForward(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim * 4),  # 4 from the paper
            nn.ReLU(),
            nn.Linear(input_dim * 4, input_dim),
            nn.Dropout(dropout_rate),
        )
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.net(input)


class Block(nn.Module):
    def __init__(self, input_dim, num_heads):
        super().__init__()
        self.attention = MultiHeadAttention(input_dim, num_heads, input_dim // num_heads)
        self.feed_forward = FeedForward(input_dim)
        self.ln1 = nn.LayerNorm(input_dim)
        self.ln2 = nn.LayerNorm(input_dim)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.attention(self.ln1(input)) + input
        xx = self.feed_forward(self.ln2(x)) + x
        return xx


# num_heads=1, block_cnt=1, embedding_dim=64 => 2.1 - 2.2
# num_heads=2, block_cnt=1, embedding_dim=64 => 2.1
# num_heads=2, block_cnt=2, embedding_dim=64 => 2.0
# 4, 4, 64 => sub 2.0
#
#
#
#
class MyGPT(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_length=128, num_heads=1, block_cnt=1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_embedding = nn.Embedding(context_length, embedding_dim)
        self.blocks = nn.Sequential(*[Block(embedding_dim, num_heads) for _ in range(block_cnt)])
        self.final_linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # (B, T) -> (B, T, C)
        B, T = input.shape
        token_embd = self.token_embedding(input) # (B, T, C)
        pos_embed = self.pos_embedding(torch.arange(T))  # (T, C)
        x = token_embd + pos_embed  # (B, T, C), matrix addition b/w 2d and 3d
        x = self.blocks(x)  # (B, T, C)
        return self.final_linear(x)


def eval(x, y, model, criterion):
    logits = model(x)

    # CrossEntropyLoss expects (N, C) and (N) to calculate loss, but we have (B, T, C) and (B, T)
    B, T, C = logits.shape
    logits = logits.reshape(-1, C)
    y = y.reshape(-1)

    loss = criterion(logits, y)
    return logits, loss


@torch.no_grad()  # no backpropagation, faster computation
def evaluate_loss(x, y, model, criterion):
    model.eval()  # necessary if dropout exists
    loss = eval(x, y, model, criterion)[1]
    model.train()
    return loss


def generate(model, input: torch.Tensor, max_tokens) -> torch.Tensor:
    context = input
    for _ in range(max_tokens):
        logits = model(context)  # (B, T, C)
        prob = F.softmax(logits[:, -1, :], dim=-1)  # (B, C)
        idx = torch.multinomial(prob, 1)  # (B, 1)
        context = torch.cat((context, idx), dim=1)
    return context

def get_parameter_sizes(model):
    total_params = 0
    for name, parameter in model.named_parameters():
        param_size = torch.prod(torch.tensor(parameter.size()))
        total_params += param_size
    print("\nTotal number of parameters:", total_params)

torch.manual_seed(42)

# process data
with open('input.txt', 'r') as f:
    input_text = f.read()
vocab_size = len(set(input_text))
ef, df = get_text_coding_func(input_text)
input_text = encode(input_text, ef)
train_to_whole_ratio = 0.9
training_input, eval_input = input_text[:int(len(input_text)*train_to_whole_ratio)], input_text[int(len(input_text)*train_to_whole_ratio):]

# model prep
embedding_dim = 64
num_heads = 2
block_cnt = 2
lr = 5e-4  # 1e-4 too slow, 1e-1 even good enough
batch_size = 32
timestep = 64
criterion = nn.CrossEntropyLoss()

max_context_length = 500
model = MyGPT(vocab_size, embedding_dim, num_heads=num_heads, block_cnt=block_cnt, context_length=max_context_length)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
num_epochs = 5000

# train
for epoch in range(num_epochs):
    x, y = get_batch(training_input, batch_size, timestep)

    logits, loss = eval(x, y, model, criterion)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch} Training Loss: Loss: {loss.item():.4f}")
        x, y = get_batch(eval_input, batch_size, timestep)
        loss = evaluate_loss(x, y, model, criterion)
        print(f"Epoch {epoch} Eval Loss: Loss: {loss.item():.4f}")

# generate text
print("Self Generated Text:")
print(decode(generate(model, torch.zeros((1, 1), dtype=torch.long), max_context_length)[0].tolist(), df))

print(get_parameter_sizes(model))
