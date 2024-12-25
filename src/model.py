import math
import torch
import torch.nn as nn

def get_FC_3layer(bin):
    return nn.Sequential(
        nn.Flatten(start_dim=-2, end_dim=-1),
        nn.Linear(bin*4, 256),
        nn.ReLU(),
        nn.Linear(256, 64),
        nn.ReLU(),
        nn.Linear(64, 2),
    )
    
class CNN(nn.Module):
    def __init__(self, data_length=20, n_channel=50, last_dense=2):
        super().__init__()
        self.z_size = data_length
        for i in range(4):
            self.z_size = self.z_size//2
        self.features = nn.Sequential(
            nn.Conv1d(n_channel, 64, kernel_size=15, padding='same'),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=15, padding='same'),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.25),
            nn.Conv1d(64, 32, kernel_size=10, padding='same'),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=10, padding='same'),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 32, kernel_size=5, padding='same'),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=5, padding='same'),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 32, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.25),
            nn.Flatten(),
            nn.Linear(32*self.z_size, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, last_dense),
        )

    def forward(self, x):
        x = x.transpose(2, 1)
        x = self.features(x)
        return x
    
###################### Transformer ######################
class TransformerClassification(nn.Module):
    def __init__(self, model_dim=386, max_seq_len=256, output_dim=2, n_layers=6, drop_prob=0):
        super().__init__()
        max_seq_len += 1
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, model_dim))
        self.ff = PositionwiseFeedForward(model_dim, 512, drop_prob)
        model_dim = 512
        self.pe = PositionalEncoder(
            model_dim=model_dim, max_seq_len=max_seq_len)
        
        self.layers = nn.ModuleList([EncoderLayer(d_model=model_dim,
                                                  ffn_hidden=model_dim,
                                                  n_head=8,
                                                  drop_prob=drop_prob)
                                     for _ in range(n_layers)])
        self.head = ClassificationHead(
            output_dim=output_dim, model_dim=model_dim)

    def forward(self, x, return_attention=False):
        cls_tokens = self.cls_token.repeat(x.shape[0], 1, 1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.ff(x)
        x = self.pe(x)
        for i, layer in enumerate(self.layers):
            x, attention = layer(x=x, src_mask=None)
            attention = attention.cpu().detach()
            if i !=0:
                attention *= attention
        logits = self.head(x)

        if return_attention:
            return logits, attention
        else:
            return logits

class EncoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x, src_mask):
        # 1. compute self attention
        _x = x
        x, attention = self.attention(q=x, k=x, v=x, mask=src_mask)
        
        # 2. add and norm
        x = self.dropout1(x)
        x = self.norm1(x + _x)
        
        # 3. positionwise feed forward network
        _x = x
        x = self.ffn(x)
      
        # 4. add and norm
        x = self.dropout2(x)
        x = self.norm2(x + _x)
        return x, attention

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.attention = ScaleDotProductAttention()
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        # 1. dot product with weight matrices
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        # 2. split tensor by number of heads
        q, k, v = self.split(q), self.split(k), self.split(v)

        # 3. do scale dot product to compute similarity
        out, attention = self.attention(q, k, v, mask=mask)
        
        # 4. concat and pass to linear layer
        out = self.concat(out)
        out = self.w_concat(out)

        # 5. visualize attention map
        # TODO : we should implement visualization

        return out, attention

    def split(self, tensor):
        batch_size, length, d_model = tensor.size()
        d_tensor = d_model // self.n_head
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)
        return tensor

    def concat(self, tensor):
        batch_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor
        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor

class ScaleDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None, e=1e-12):
        # input is 4 dimension tensor
        # [batch_size, head, length, d_tensor]
        batch_size, head, length, d_tensor = k.size()

        # 1. dot product Query with Key^T to compute similarity
        k_t = k.transpose(2, 3)  # transpose
        score = (q @ k_t) / math.sqrt(d_tensor)  # scaled dot product

        # 2. apply masking (opt)
        if mask is not None:
            score = score.masked_fill(mask == 0, -10000)

        # 3. pass them softmax to make [0, 1] range
        # _score = score
        score = self.softmax(score)

        # 4. multiply with Value
        v = score @ v

        return v, score
    
class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        # '-1' means last dimension. 

        out = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * out + self.beta
        return out


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, hidden, drop_prob=0):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class PositionalEncoder(nn.Module):
    def __init__(self, model_dim=300, max_seq_len=256):
        super().__init__()
        self.model_dim = model_dim
        pe = torch.zeros(max_seq_len, model_dim)

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        pe = pe.to(device)

        for pos in range(max_seq_len):
            for i in range(0, model_dim, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / model_dim)))
                pe[pos, i + 1] = math.cos(pos /
                                          (10000 ** ((2 * (i + 1)) / model_dim)))

        self.pe = pe.unsqueeze(0)
        self.pe.requires_grad = False

    def forward(self, x):
        result = x + self.pe
        return result     

class ClassificationHead(nn.Module):
    def __init__(self, model_dim=300, output_dim=2):
        super().__init__()

        self.linear = nn.Linear(model_dim, output_dim)

    def forward(self, x):
        x0 = x[:, 0, :]
        out = self.linear(x0)

        return out
    
###################### CvT ######################
class CvT(nn.Module):
    def __init__(self, ch, length, model_dim=64, output_dim=2, n_layers=6, bin=4, drop_prob=0):
        super().__init__()

        self.ch, self.length = ch, length
        # self.model_dim, self.output_dim = model_dim, output_dim
        self.model_dim, self.output_dim = 512//bin, output_dim
        self.n_layers, self.bin = n_layers, bin

        if (self.length % self.bin) == 0:
            self.tf_length = (self.length // self.bin)
            pad = 0
        else: 
            self.tf_length = (self.length // self.bin) + 1
            pad = self.tf_length*self.bin - self.length
        self.padding = nn.ZeroPad2d((0, 0, 0, pad))

        self.cls_token = nn.Parameter(torch.randn(1, 1, self.model_dim*self.bin))
        
        self.cnn = nn.Sequential(
            nn.Conv1d(self.ch, 16, kernel_size=3, padding='same'),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding='same'),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, self.model_dim, kernel_size=3, padding='same'),
            nn.BatchNorm1d(self.model_dim),
        )

        self.pe = PositionalEncoder(
            model_dim=self.model_dim*self.bin, max_seq_len=self.tf_length+1)

        self.tf_layers = nn.ModuleList([EncoderLayer(d_model=self.model_dim*self.bin,
                                                  ffn_hidden=self.model_dim*self.bin,
                                                  n_head=8,
                                                  drop_prob=drop_prob)
                                                for _ in range(self.n_layers)])

        self.head = ClassificationHead(
            model_dim=self.model_dim*self.bin, output_dim=self.output_dim)

    def forward(self, x, return_attention=False):
        x = self.padding(x)

        # --- CNN ---
        x = x.transpose(2, 1)
        x = self.cnn(x)
        x = x.transpose(2, 1)

        # --- Transformer ---
        x = x.reshape(x.size(0), self.tf_length, -1)

        cls_tokens = self.cls_token.repeat(x.size(0), 1, 1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pe(x)
        
        for i, layer in enumerate(self.tf_layers):
            x, attention = layer(x=x, src_mask=None)
            attention = attention.cpu().detach()
            if i !=0:
                attention *= attention
        logits = self.head(x)

        if return_attention:
            return logits, attention
        else:
            return logits