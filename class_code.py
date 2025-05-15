
# This code is a simple implementation of a character-level GPT for text generation using PyTorch.
import requests
import os
import torch
import torch.nn as nn
import torch.optim as optim 
from torch.nn import functional as F

batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2


torch.manual_seed(1337)

# download the tiny shakespeare dataset
text_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
if not os.path.exists(text_file_path):
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    with open(text_file_path, 'w') as f:
        f.write(requests.get(data_url).text)

with open(text_file_path, 'r') as f:
    data = f.read()

char = sorted(list(set(data)))
vocab_size = len(char)

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(char) }
itos = { i:ch for i,ch in enumerate(char) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# create the train and test splits
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# encode both to integers
def get_batch(split):
    data = train_data if split == 'train' else val_data
    # 生成batch_size个随机起始位置，因为后续要取block_size个字符，所以不能超过len(data) - block_size，否则会越界
    ix = torch.randint(len(data) - block_size, (batch_size,))
    # 将batch_size个编码后的数字堆叠成一个tensor
    x = torch.stack([torch.tensor(encode(data[i:i+block_size])) for i in ix])
    y = torch.stack([torch.tensor(encode(data[i+1:i+block_size+1])) for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss(model, split):
    model.eval()
    out = {}
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# B -> batch size, T -> time steps(block_size), C -> channels

class Header(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) # 下三角矩阵
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x) # (B,T,head_size)
        q = self.query(x) # (B,T,head_size)
        v = self.value(x) # (B,T,head_size)

        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B,T,C) @ (B,C,T) -> (B,T,T)
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf')) # 只保留下三角部分  (B,T,T)
        wei = F.softmax(wei, dim=-1) # softmax归一化       (B,T,T)
        wei = self.dropout(wei)     # (B,T,T)

        out = wei @ v # (B,T,T) @ (B,T,head_size) -> (B,T,head_size)
        return out
    
class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Header(head_size) for _ in range(n_head)]) # 多头自注意力
        self.proj = nn.Linear(n_embd, n_embd) # 线性变换
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1) # (B,T,C)
        out = self.proj(self.dropout(out)) # (B,T,C)
        return out

# FeedForward 层让模型不仅能“交流信息”，还能“加工信息”，是提升模型性能和表达力的重要结构
# 加入这个之前，计算logits太快了，模型没有时间去“思考”，所以加入了这个
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd * 4),
            nn.ReLU(),
            nn.Linear(n_embd * 4, n_embd),
            nn.Dropout(dropout) 
        )

    def forward(self, x):
        return self.net(x)      # (B,T,C)
    
class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()  
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd) # 层归一化
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))  # 残差连接
        return x        # (B,T,C)

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # nn.Embedding用于将离散的整数索引（如字符或单词的编号）映射为连续的高维向量
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)]) # 堆叠多个Block)
        self.ln_f = nn.LayerNorm(n_embd) # 层归一化
        self.lm_head = nn.Linear(n_embd, vocab_size) # 线性层用于将嵌入向量映射回词汇表大小的输出

    def forward(self, idx, targets=None):
        token_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(idx.shape[1], device=device))
        x = token_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C) # (B*T,C)
            targets = targets.view(B*T) # (B*T,)
            loss = F.cross_entropy(logits, targets) # 交叉熵损失函数

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:] 
            logits, loss = self(idx_cond)        # (B,T,C)
            # 因为我们只需要预测下一个字符，所以只取最后一个时间步的输出
            logits = logits[:, -1, :]       # (B,C)
            probs = F.softmax(logits, dim=-1) # (B,C)
            idx_next = torch.multinomial(probs, num_samples=1)      # (B,1)
            idx = torch.cat((idx, idx_next), dim=1)     # (B,T+1)
        return idx
    
model = BigramLanguageModel()
model.to(device)
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss(model, 'train')
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    X, Y = get_batch('train')
    logits, loss = model(X, Y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate some text
context = 'hello'
context = torch.tensor(encode(context), dtype=torch.long, device=device)[None, :] # (1,T)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))





