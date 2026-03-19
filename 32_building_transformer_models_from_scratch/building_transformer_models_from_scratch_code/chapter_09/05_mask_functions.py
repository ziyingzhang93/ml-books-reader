import torch

def create_causal_mask(seq_len):
    """
    Create a causal mask for autoregressive attention.

    Args:
        seq_len: Length of the sequence

    Returns:
        Causal mask of shape (seq_len, seq_len)
    """
    mask = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1)
    return mask

def create_padding_mask(batch, padding_token_id):
    """
    Create a padding mask for a batch of sequences.

    Args:
        batch: Batch of sequences, shape (batch_size, seq_len)
        padding_token_id: ID of the padding token

    Returns:
        Padding mask of shape (batch_size, seq_len, seq_len)
    """
    batch_size, seq_len = batch.shape
    padded = torch.zeros_like(batch).float() \
                  .masked_fill(batch == padding_token_id, float('-inf'))
    mask = torch.zeros(batch_size, seq_len, seq_len) + padded[:,:,None] + padded[:,None,:]
    return mask[:, None, :, :]

print(create_causal_mask(5))
batch = torch.tensor([
    [1, 2, 3, 4, 5, 6],
    [1, 2, 3, 0, 0, 0],
    [1, 2, 3, 4, 0, 0]
])
print(create_padding_mask(batch, 0))
