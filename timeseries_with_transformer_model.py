import torch
import torch.nn as nn
import lightning as L

class PositionalEncoding(nn.Module):
    def __init__(self, d_model:int, max_len:int):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

class TimeSeriesTransformer(L.LightningModule):
    def __init__(
        self, 
        input_dim:int, 
        d_model:int, 
        nhead:int, 
        num_encoder_layers:int, 
        num_decoder_layers:int, 
        dim_feedforward:int, 
        dropout:float, 
        sequences:list[tuple[int, list[str]]],
        number_pred_features:int):
        
        super(TimeSeriesTransformer, self).__init__()

        self.number_pred_features = number_pred_features

        # Calculate the total sequence length by summing all the history lengths
        total_seq_len = sum(history_len for history_len, _ in sequences)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        
        self.input_embedding = nn.Sequential(
            nn.Linear(input_dim, d_model),
            # nn.ReLU(),
            # nn.Linear(d_model, d_model)
        )
        
        self.fc_out = nn.Sequential(
            nn.Linear(d_model, self.number_pred_features),
            # nn.ReLU(),
            # nn.Linear(d_model, self.number_pred_features)  
        )
        
        self.positional_encoding = PositionalEncoding(d_model, max_len=total_seq_len)
        
        # self.tgt_mask = None

    def forward(self, src, tgt):
        
        """
        len_src = len(src)
        
        if self.tgt_mask is None or self.tgt_mask.size(0) != len_src:
            device = src.device
            mask = nn.Transformer.generate_square_subsequent_mask(len_src).to(device)
            self.tgt_mask = mask        
        """
        
        src = self.input_embedding(src)
        tgt = self.input_embedding(tgt)
        src = self.positional_encoding(src)
        tgt = self.positional_encoding(tgt)

        output = self.transformer(
            src, tgt,
            #tgt_mask=self.tgt_mask
        )
        output = self.fc_out(output[:, -1, :])  # Use the last time step output for prediction
        output = output.view(-1, self.number_pred_features)  # Reshape to (batch_size, num_features)
        return output

    def training_step(self, batch, batch_idx):
        src, tgt, y = batch
        output = self(src, tgt)
        loss = nn.MSELoss()(output, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        src, tgt, y = batch

        output = self(src, tgt)
        loss = nn.MSELoss()(output, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return {"val_loss": loss}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        src = batch  # Assuming batch contains only the source tensor      
        
        # Embed and add positional encoding to source
        src = self.input_embedding(src)
        src = self.positional_encoding(src)

        # Encode the source
        memory = self.transformer.encoder(src)

        # Initialize prediction tensor
        batch_size = memory.size(1)
        pred = torch.zeros(batch_size, self.number_pred_features).to(memory.device)

        # Initial input token (e.g., start of sequence token)
        tgt = self.input_embedding(pred.unsqueeze(0))
        tgt = self.positional_encoding(tgt)

        # Iteration based on the window_size or any predefined steps
        for _ in range(8):  
            output = self.transformer.decoder(tgt, memory)
            output = self.fc_out(output[-1, :, :])
            pred = output.view(-1, self.number_pred_features)
            tgt = torch.cat([tgt, self.input_embedding(pred.unsqueeze(0))], dim=0)

        return pred    
    