import os
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tokenizers
import tqdm

# Download novels from Project Gutenberg
DATASOURCE = {
    "moby_dick": "https://www.gutenberg.org/ebooks/2701.txt.utf-8",
    "frankenstein": "https://www.gutenberg.org/ebooks/84.txt.utf-8",
    "dracula": "https://www.gutenberg.org/ebooks/345.txt.utf-8",
    "little_women": "https://www.gutenberg.org/ebooks/37106.txt.utf-8",
    "pride_and_prejudice": "https://www.gutenberg.org/ebooks/1342.txt.utf-8",
    "alice_in_wonderland": "https://www.gutenberg.org/ebooks/11.txt.utf-8",
    "crime_and_punishment": "https://www.gutenberg.org/ebooks/2554.txt.utf-8",
    "tom_sawyer": "https://www.gutenberg.org/ebooks/74.txt.utf-8",
    "tale_of_two_cities": "https://www.gutenberg.org/ebooks/98.txt.utf-8",
    "sherlock_holmes": "https://www.gutenberg.org/ebooks/1661.txt.utf-8",
    "war_and_peace": "https://www.gutenberg.org/ebooks/2600.txt.utf-8",
}
for filename, url in DATASOURCE.items():
    if not os.path.exists(f"{filename}.txt"):
        response = requests.get(url)
        with open(f"{filename}.txt", "wb") as f:
            f.write(response.content)

# Read and preprocess the text
def preprocess_gutenberg(filename):
    with open(filename, "r", encoding="utf-8") as f:
        text = f.read()

    # Find the start and end of the actual content
    start = text.find("*** START OF THE PROJECT GUTENBERG EBOOK")
    start = text.find("\n", start) + 1
    end = text.find("*** END OF THE PROJECT GUTENBERG EBOOK")

    # Extract the main content
    text = text[start:end].strip()

    # Basic preprocessing
    # Remove multiple newlines and spaces
    text = "\n".join(line.strip() for line in text.split("\n") if line.strip())
    return text

def get_dataset_text():
    all_text = []
    for filename in DATASOURCE:
        text = preprocess_gutenberg(f"{filename}.txt")
        all_text.append(text)
    return all_text

# Tokenization with BPE
if os.path.exists("gutenberg_tokenizer.json"):
    tokenizer = tokenizers.Tokenizer.from_file("gutenberg_tokenizer.json")
else:
    tokenizer = tokenizers.Tokenizer(tokenizers.models.BPE())
    # Configure pre-tokenizer add space at beginning of the sentence
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=True)
    # Configure decoder so that the boundary symbols will be removed
    tokenizer.decoder = tokenizers.decoders.ByteLevel()
    # Train BPE
    VOCAB_SIZE = 10000
    trainer = tokenizers.trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=["[pad]", "[eos]"],
        show_progress=True
    )
    text = get_dataset_text()
    tokenizer.train_from_iterator(text, trainer=trainer)
    tokenizer.enable_padding(pad_id=tokenizer.token_to_id("[pad]"), pad_token="[pad]")
    # Save the trained tokenizer
    tokenizer.save("gutenberg_tokenizer.json", pretty=True)

# Create PyTorch dataset
class GutenbergDataset(torch.utils.data.Dataset):
    def __init__(self, text, tokenizer, seq_len=512):
        self.seq_len = seq_len
        # Encode the entire text
        self.encoded = tokenizer.encode(text).ids

    def __len__(self):
        return len(self.encoded) - self.seq_len

    def __getitem__(self, idx):
        chunk = self.encoded[idx:idx + self.seq_len + 1]  # +1 for target
        x = torch.tensor(chunk[:-1])
        y = torch.tensor(chunk[1:])
        return x, y

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(x, cos, sin):
    return (x * cos) + (rotate_half(x) * sin)

class RotaryPositionalEncoding(nn.Module):
    def __init__(self, dim, max_seq_len=1024):
        super().__init__()
        N = 10000
        inv_freq = 1. / (N ** (torch.arange(0, dim, 2).float() / dim))
        position = torch.arange(max_seq_len).float()
        inv_freq = torch.cat((inv_freq, inv_freq), dim=-1)
        sinusoid_inp = torch.outer(position, inv_freq)
        self.register_buffer("cos", sinusoid_inp.cos())
        self.register_buffer("sin", sinusoid_inp.sin())

    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.size(1)
        cos = self.cos[:seq_len].view(1, seq_len, 1, -1)
        sin = self.sin[:seq_len].view(1, seq_len, 1, -1)
        return apply_rotary_pos_emb(x, cos, sin)

class SwiGLU(nn.Module):
    def __init__(self, hidden_dim, intermediate_dim):
        super().__init__()
        self.gate = nn.Linear(hidden_dim, intermediate_dim)
        self.up = nn.Linear(hidden_dim, intermediate_dim)
        self.down = nn.Linear(intermediate_dim, hidden_dim)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.act(self.gate(x)) * self.up(x)
        x = self.down(x)
        return x

class GQA(nn.Module):
    def __init__(self, hidden_dim, num_heads, num_kv_heads=None, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = hidden_dim // num_heads
        self.num_groups = num_heads // num_kv_heads
        self.dropout = dropout
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, q, k, v, mask=None, rope=None):
        q_batch_size, q_seq_len, hidden_dim = q.shape
        k_batch_size, k_seq_len, hidden_dim = k.shape
        v_batch_size, v_seq_len, hidden_dim = v.shape

        # projection
        q = self.q_proj(q).view(q_batch_size, q_seq_len, -1, self.head_dim).transpose(1, 2)
        k = self.k_proj(k).view(k_batch_size, k_seq_len, -1, self.head_dim).transpose(1, 2)
        v = self.v_proj(v).view(v_batch_size, v_seq_len, -1, self.head_dim).transpose(1, 2)

        # apply rotary positional encoding
        if rope:
            q = rope(q)
            k = rope(k)

        # compute grouped query attention
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        output = F.scaled_dot_product_attention(q, k, v,
                                                attn_mask=mask,
                                                dropout_p=self.dropout,
                                                enable_gqa=True)
        output = output.transpose(1, 2).reshape(q_batch_size, q_seq_len, hidden_dim).contiguous()
        output = self.out_proj(output)
        return output

class DecoderLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, num_kv_heads, dropout=0.1):
        super().__init__()
        self.self_attn = GQA(hidden_dim, num_heads, num_kv_heads, dropout)
        self.mlp = SwiGLU(hidden_dim, 4 * hidden_dim)
        self.norm1 = nn.RMSNorm(hidden_dim)
        self.norm2 = nn.RMSNorm(hidden_dim)

    def forward(self, x, mask=None, rope=None):
        # self-attention sublayer
        out = self.norm1(x)
        out = self.self_attn(out, out, out, mask, rope)
        x = out + x
        # MLP sublayer
        out = self.norm2(x)
        out = self.mlp(out)
        return out + x

class TextGenerationModel(nn.Module):
    def __init__(self, num_layers, num_heads, num_kv_heads, hidden_dim,
                 max_seq_len, vocab_size, dropout=0.1):
        super().__init__()
        self.rope = RotaryPositionalEncoding(hidden_dim // num_heads, max_seq_len)
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.decoders = nn.ModuleList([
            DecoderLayer(hidden_dim, num_heads, num_kv_heads, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.RMSNorm(hidden_dim)
        self.out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, ids, mask=None):
        x = self.embedding(ids)
        for decoder in self.decoders:
            x = decoder(x, mask, self.rope)
        x = self.norm(x)
        return self.out(x)

def create_causal_mask(seq_len, device):
    """Create a causal mask for autoregressive attention."""
    mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=device), diagonal=1)
    return mask

# Training configuration
model_config = {
    "num_layers": 8,
    "num_heads": 8,
    "num_kv_heads": 4,
    "hidden_dim": 768,
    "max_seq_len": 512,
    "vocab_size": len(tokenizer.get_vocab()),
    "dropout": 0.1,
}

# Initialize model, optimizer, etc.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TextGenerationModel(**model_config).to(device)

# Create dataset and dataloader
BATCH_SIZE = 32
text = "\n".join(get_dataset_text())
dataset = GutenbergDataset(text, tokenizer, seq_len=model_config["max_seq_len"])
dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Training loop
if os.path.exists("textgen_model.pth"):
    model.load_state_dict(torch.load("textgen_model.pth"))
else:
    N_EPOCHS = 2
    LR = 0.0005
    WARMUP_STEPS = 2000
    CLIP_NORM = 6.0

    optimizer = optim.AdamW(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id("[pad]"))

    # Learning rate scheduling
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=WARMUP_STEPS)
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=N_EPOCHS * len(dataloader) - WARMUP_STEPS, eta_min=0)
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[WARMUP_STEPS])

    print(f"Training for {N_EPOCHS} epochs with {len(dataloader)} steps per epoch")
    best_loss = float('inf')

    for epoch in range(N_EPOCHS):
        model.train()
        epoch_loss = 0

        progress_bar = tqdm.tqdm(dataloader, desc=f"Epoch {epoch+1}/{N_EPOCHS}")
        for x, y in progress_bar:
            x = x.to(device)
            y = y.to(device)

            # Create causal mask
            mask = create_causal_mask(x.shape[1], device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(x, mask.unsqueeze(0))

            # Compute loss
            loss = loss_fn(outputs.view(-1, outputs.shape[-1]), y.view(-1))

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), CLIP_NORM, error_if_nonfinite=True
            )
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()

            # Show loss in tqdm
            progress_bar.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{N_EPOCHS}; Avg loss: {avg_loss:.4f}")

        # Save checkpoint if loss improved
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "textgen_model.pth")

# Generation function
def generate_text(model, tokenizer, prompt, max_length=100, temperature=0.7):
    model.eval()
    device = next(model.parameters()).device

    # Encode the prompt
    input_ids = torch.tensor(tokenizer.encode(prompt).ids).unsqueeze(0).to(device)

    with torch.no_grad():
        for _ in range(max_length):
            # Get model predictions for the next token as the last element of the output
            outputs = model(input_ids)
            next_token_logits = outputs[:, -1, :] / temperature
            # Sample from the distribution
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            # Append to input_ids
            input_ids = torch.cat([input_ids, next_token], dim=1)
            # Stop if we predict the end token
            if next_token[0].item() == tokenizer.token_to_id("[eos]"):
                break

    return tokenizer.decode(input_ids[0].tolist())

# Test the model with some prompts
test_prompts = [
    "Once upon a time,",
    "We the people of the",
    "In the beginning was the",
]

print("\nGenerating sample texts:")
for prompt in test_prompts:
    generated = generate_text(model, tokenizer, prompt)
    print(f"\nPrompt: {prompt}")
    print(f"Generated: {generated}")
    print("-" * 80)
