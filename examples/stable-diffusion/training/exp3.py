import torch
import sys
import torch.nn.functional as F

#  def forward(self, primals_1: "Sym(s0)", view_3: "f32[16, 4096, 40]", view_4: "f32[16, 40, s0]", _safe_softmax: "f32[2, 8, 4096, s0]",
#  view_6: "f32[16, 4096, s0]", view_7: "f32[16, s0, 40]", tangents_1: "bf16[2, 4096, 320]"):

def foo(query, key, value, attention_mask=None):
    batch_size = 2
    attn_heads = 8 #key.shape[-2]
    head_dim = 40 #inner_dim // attn_heads

    query = query.view(batch_size, -1, attn_heads, head_dim).transpose(1, 2)
    key = key.view(batch_size, -1, attn_heads, head_dim).transpose(1, 2)
    value = value.view(batch_size, -1, attn_heads, head_dim).transpose(1, 2)

    hidden_states = F.scaled_dot_product_attention(
        query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
    )

    hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn_heads * head_dim)
    return hidden_states

device = 'hpu'
fn = torch.compile(foo, backend='hpu_backend')

query_base_shape = [2, 4096, 8, 40]
kv_base_shape = [2, 77, 8, 40]

query = torch.randn(query_base_shape, requires_grad=True, device=device)
key = torch.randn(kv_base_shape, requires_grad=True, device=device)
value = torch.randn(kv_base_shape, requires_grad=True, device=device)

y = fn(query, key, value)
z = y.sum()
z.backward()
print(f'{y.shape=}')
sys.exit(0)