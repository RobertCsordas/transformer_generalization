import torch
import torch.nn
import torch.nn.functional as F
from typing import Optional
from .multi_head_attention import AttentionMask, MultiHeadAttentionBase
import framework
import math


class RelativeAttentionBase(MultiHeadAttentionBase):
    def __init__(self, state_size: int, n_heads: int, dropout: float):
        super().__init__(state_size, n_heads, dropout=dropout)

    def _shift(self, posmat: torch.Tensor) -> torch.Tensor:
        # Slice out a matrix diagonally. Each successive row is sliced one position to the left compared.
        # shape: [n_batch, n_head, n_out, n_in * 2 - 1]
        # return: [n_batch, n_head, n_out, n_in]
        p = F.pad(posmat, (0, 1, 0, 1)).flatten(-2)  # [n_batch, n_head, (n_out + 1) * n_in * 2]
        p = p.narrow(-1, posmat.shape[-1] // 2, posmat.shape[-1] * posmat.shape[-2]).view_as(posmat)

        return p.narrow(-1, 0, (posmat.shape[-1] + 1) // 2)

    def _attention(self, mask: Optional[torch.Tensor],
                   q_content: torch.Tensor, k_content: torch.Tensor,
                   q_pos: torch.Tensor, k_pos: torch.Tensor,
                   v: torch.Tensor) -> torch.Tensor:

        # shape of q_content, q_pos, k_pos: [n_batch * n_heads, n_steps, data_size]
        # k_pos: [n_heads, n_in * 2 - 1, data_size]
        # Output shape [n_batch * n_heads, n_out, data_size]

        n_batch = q_content.shape[0] // self.n_heads
        n_out_steps = q_content.shape[1]

        # content-content addressing
        content = torch.bmm(q_content, k_content.transpose(1, 2))

        # content-pos addressing.
        pos = torch.matmul(q_pos.view(n_batch, self.n_heads, n_out_steps, -1), k_pos.transpose(-1, -2))  # [n_batch, n_head, n_out, n_in * 2 - 1]
        pos = self._shift(pos).flatten(0, 1)

        # Logits shape: [n_batch * n_heads, n_out, n_in]
        return self._attention_read(mask, content + pos, v)

    def _get_pos_subset(self, pos_encoding: torch.Tensor, length: int, offset: int) -> torch.Tensor:
        l_slice = 2 * length - 1
        assert pos_encoding.shape[0] > l_slice
        return pos_encoding.narrow(0, pos_encoding.shape[0] // 2 - length + 1 - offset, 2 * length - 1)


class FixedRelativeMultiheadAttention(RelativeAttentionBase):
    def __init__(self, state_size: int, n_heads: int, dropout: float = 0.0, global_pos_bias: bool = True,
                 global_content_bias: bool = True, input_size: Optional[int] = None):
        super().__init__(state_size, n_heads, dropout)

        self.data_to_kv = torch.nn.Linear(state_size, 2 * n_heads * self.projection_size, bias=False)
        self.data_to_q = torch.nn.Linear(state_size if input_size is None else input_size,
                                         n_heads * self.projection_size, bias=False)

        self.global_content_bias = torch.nn.Parameter(torch.zeros([n_heads, self.projection_size])) \
                                   if global_content_bias else None
        self.global_pos_bias = torch.nn.Parameter(torch.zeros([n_heads, self.projection_size])) \
                               if global_pos_bias else None

        self.pos_to_pq = torch.nn.Linear(state_size, self.n_heads * self.projection_size, bias=False)
        self.register_buffer("pos_encoding", self._create_buffer(1000))

    def _create_buffer(self, max_len: int):
        return framework.layers.sinusoidal_pos_embedding(self.state_size, 2 * max_len - 1, -max_len + 1,
                                                         device=self.data_to_q.weight.device)

    def get_pos(self, l: int, offset: int) -> torch.Tensor:
        if self.pos_encoding.shape[0] < 2 * (l + offset) - 1:
            self.pos_encoding = self._create_buffer(int(2**math.ceil(math.log2(2 * (l + offset) - 1))))

        return self._get_pos_subset(self.pos_encoding, l, offset)

    def add_head_specific_bias(self, data: torch.Tensor, bias: Optional[torch.Tensor]) -> torch.Tensor:
        # data [batch * n_heads, len, c]
        # bias [n_heads, c]
        return (data.view(-1, bias.shape[0], *data.shape[1:]) + bias.unsqueeze(1).type_as(data)).view_as(data) \
               if bias is not None else data

    def forward(self, curr_state: torch.Tensor, attend_to: torch.Tensor, mask: Optional[AttentionMask],
                pos_offset: int = 0, need_weights: bool = False):
        # curr_state: [batch_size, out_len, c]
        # attend_to: [batch_size, in_len, c]
        batch_size, in_len = attend_to.shape[0:2]
        out_len = curr_state.shape[1]

        k_content, v = self.transform_data(attend_to, self.data_to_kv, 2)
        q, = self.transform_data(curr_state, self.data_to_q, 1)

        k_pos = self.pos_to_pq(self.get_pos(in_len, pos_offset)).view(-1, self.n_heads, self.projection_size).\
                transpose(0, 1)  # n_heads, 2*in_len -1 , projection_size

        q_content = self.add_head_specific_bias(q, self.global_content_bias)
        q_pos = self.add_head_specific_bias(q, self.global_pos_bias)

        data, scores = self.merged_attention(batch_size, out_len, mask, q_content, k_content, q_pos, k_pos, v,
                                             need_weights=need_weights)
        
        if need_weights:
            # Calculate the mean over the heads
            return data, scores.mean(1)
        else:
            return data

    def reset_parameters(self):
        super().reset_parameters()

        torch.nn.init.xavier_uniform_(self.data_to_q.weight)
        torch.nn.init.xavier_uniform_(self.pos_to_pq.weight)
        torch.nn.init.xavier_uniform_(self.data_to_kv.weight[:self.data_to_kv.weight.shape[0]//2])
        torch.nn.init.xavier_uniform_(self.data_to_kv.weight[self.data_to_kv.weight.shape[0]//2:])

        if self.global_content_bias is not None:
            self.global_content_bias.fill_(0)

        if self.global_pos_bias is not None:
            self.global_pos_bias.fill_(0)
