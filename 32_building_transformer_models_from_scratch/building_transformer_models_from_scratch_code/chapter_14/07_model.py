import torch
import torch.nn as nn

class Seq2SeqLSTM(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_seq, target_seq):
        batch_size, target_len = target_seq.shape
        outputs = []
        _enc_out, hidden, cell = self.encoder(input_seq)
        dec_in = target_seq[:, :1]
        for t in range(target_len-1):
            pred, hidden, cell = self.decoder(dec_in, hidden, cell)
            pred = pred[:, -1:, :]
            outputs.append(pred)
            dec_in = torch.cat([dec_in, pred.argmax(dim=2)], dim=1)
        outputs = torch.cat(outputs, dim=1)
        return outputs
