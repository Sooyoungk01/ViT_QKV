import torch 
import torch.nn as nn

import torch.cuda.nvtx as nvtx

class SelfAttention(nn.Module):
    def __init__(
        self,
        embedding_dims,
        heads,
        dropout
    ):
        super(SelfAttention, self).__init__()
        self.heads = heads
        self.embedding_dims = embedding_dims
        self.head_dims = int(embedding_dims/heads)

        self.key = nn.Linear(self.head_dims, self.head_dims)
        self.query = nn.Linear(self.head_dims, self.head_dims)
        self.value = nn.Linear(self.head_dims, self.head_dims)

        self.fc = nn.Linear(self.head_dims*self.heads, self.embedding_dims)

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key, value, mask):
        Batch = query.shape[0]

        query_len, key_len, value_len = query.shape[1], key.shape[1], value.shape[1]

        query = query.reshape(Batch, query_len, self.heads, self.head_dims)
        key = key.reshape(Batch, key_len, self.heads, self.head_dims)
        value = value.reshape(Batch, value_len, self.heads, self.head_dims)

        #change
        nvtx.range_push("Q")
        query = query.permute(2, 0, 1, 3)
        query_total = torch.cat([query[0], query[1], query[2], query[3]], dim=1)
        query_total = self.query(query_total)
        splits_q = query_total.split(5, dim=1)
        query = torch.stack([splits_q[0], splits_q[1], splits_q[2], splits_q[3]], dim=2)
        nvtx.range_pop()
        
        nvtx.range_push("K")
        key = key.permute(2, 0, 1, 3)
        key_total = torch.cat([key[0], key[1], key[2], key[3]], dim=1)
        key_total = self.key(key_total) 
        splits_k = key_total.split(5, dim=1)
        key = torch.stack([splits_k[0], splits_k[1], splits_k[2], splits_k[3]], dim=2)
        nvtx.range_pop()
        
        nvtx.range_push("V")
        value = value.permute(2, 0, 1, 3)
        value_total = torch.cat([value[0], value[1], value[2], value[3]], dim=1)
        value_total = self.value(value_total) 
        splits_v = value_total.split(5, dim=1)
        value = torch.stack([splits_v[0], splits_v[1], splits_v[2], splits_v[3]], dim=2)
        nvtx.range_pop()


        attention_score = torch.einsum('bqhd,bkhd->bhqk', [query, key])

        if mask is not None:
            attention_score = attention_score.masked_fill(mask==0, float('-1e20'))

        attention_score = attention_score/((self.head_dims)**(1/2))
        nvtx.range_push("softmax")
        attention_score = torch.softmax(attention_score, dim=-1)
        nvtx.range_pop()
        out = torch.einsum('bhqv,bvhd->bqhd', [attention_score, value]).reshape(
            Batch, query_len, self.heads*self.head_dims
        )

        out = self.dropout(self.fc(out))

        return out



class TransformerBlock(nn.Module):
    def __init__(
        self,
        embedding_dims,
        heads,
        dropout, 
        forward_expansion,
        layer_norm_eps
    ):
        super(TransformerBlock, self).__init__()
        self.layer_norm1 = nn.LayerNorm(embedding_dims, eps=layer_norm_eps)
        self.layer_norm2 = nn.LayerNorm(embedding_dims, eps=layer_norm_eps)
        self.attention = SelfAttention(embedding_dims, heads, dropout)
        self.feed_forward = nn.Sequential(
                nn.Linear(embedding_dims, embedding_dims*forward_expansion),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embedding_dims*forward_expansion, embedding_dims),
                nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask):
        nvtx.range_push("norm")
        norm = self.layer_norm1(x)
        nvtx.range_pop()
        nvtx.range_push("attention")
        attention_block = self.attention(norm, norm, norm, mask)
        nvtx.range_pop()
        add = x + attention_block
        nvtx.range_push("norm")
        norm = self.layer_norm2(add)
        nvtx.range_pop()
        nvtx.range_push("feed forward")
        
        feed_forward = self.feed_forward(norm)
        nvtx.range_pop()
        out = feed_forward + add
        return out


class ViT(nn.Module):
    def __init__(
        self,
        patch_height,
        patch_width,
        max_len,
        embedding_dims,
        heads,
        forward_expansion,
        num_layers,
        dropout,
        layer_norm_eps,
        num_classes
    ):
        super(ViT, self).__init__()
        
        self.vit_blocks = nn.Sequential(
            *[
                TransformerBlock(
                    embedding_dims,
                    heads,
                    dropout,
                    forward_expansion,
                    layer_norm_eps
                )
                for _ in range(num_layers)
            ]
            
        )
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.cls_embedding = nn.Parameter(torch.zeros(1, 1, embedding_dims))
        self.patch_embeddings = nn.Linear(embedding_dims, embedding_dims)
        self.postional_embedding = nn.Parameter(torch.zeros(1, max_len+1, embedding_dims))
        self.to_cls_token = nn.Identity()
        self.classifier = nn.Sequential(
            nn.LayerNorm(embedding_dims),
            nn.Linear(embedding_dims, num_classes*4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(num_classes*4, num_classes)
        )
        self.dropout = nn.Dropout(dropout)


    def forward(self, images):
        patches = images.unfold(2, self.patch_height, self.patch_width).unfold(3, self.patch_height, self.patch_width)
        patches = patches.permute(0, 2, 3, 1, 4, 5)
        patches = patches.reshape(
            patches.shape[0],
            patches.shape[1],
            patches.shape[2],
            patches.shape[3]*patches.shape[4]*patches.shape[5]
        )
        patches = patches.view(patches.shape[0], -1, patches.shape[-1])

        nvtx.range_push("cls_embedding")
        x = self.cls_embedding.expand(patches.shape[0], -1, -1)
        nvtx.range_pop()
        nvtx.range_push("patch_embedding")
        patch_embeddings = self.patch_embeddings(patches)
        nvtx.range_pop()
        x = torch.cat((x, patch_embeddings), dim=1) + self.postional_embedding
        x = self.dropout(x)
        mask = None
        for block in self.vit_blocks:
            nvtx.range_push("decoder")
            x = block(x, mask)
            nvtx.range_pop()
        nvtx.range_push("classifier")
        out = self.to_cls_token(x[:, 0])
        out = self.classifier(out)
        nvtx.range_pop()
        return out


if __name__ == "__main__":

    model = ViT(
        patch_height = 16,
        patch_width = 16,
        embedding_dims = 768,
        dropout = 0.1,
        heads = 4,
        num_layers = 4,
        forward_expansion = 4,
        max_len = int((32*32)/(16*16)),
        layer_norm_eps = 1e-5,
        num_classes = 10,
    )

    a = torch.randn(32, 3, 32, 32)
    output = model(a)
    print(output.shape)
