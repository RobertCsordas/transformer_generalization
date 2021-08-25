import torch
import torch.nn
import torch.nn.functional as F
import framework
from layers import Transformer, TiedEmbedding
from typing import Callable, Optional
import math


# Cannot be dataclass, because that won't work with gather
class TransformerResult(framework.data_structures.DotDict):
    data: torch.Tensor
    length: torch.Tensor

    @staticmethod
    def create(data: torch.Tensor, length: torch.Tensor):
        return TransformerResult({"data": data, "length": length})


class TransformerEncDecModel(torch.nn.Module):
    def __init__(self, n_input_tokens: int, n_out_tokens: int, state_size: int = 512, ff_multiplier: float = 4,
                 max_len: int=5000, transformer = Transformer, tied_embedding: bool=False,
                 pos_embeddig: Optional[Callable[[torch.Tensor, int], torch.Tensor]] = None,
                 encoder_sos: bool = True, same_enc_dec_embedding: bool = False, embedding_init: str = "pytorch",
                 in_embedding_size: Optional[int] = None, out_embedding_size: Optional[int] = None, 
                 scale_mode: str = "none", **kwargs):
        '''
        Transformer encoder-decoder.

        :param n_input_tokens: Number of channels for the input vectors
        :param n_out_tokens: Number of channels for the output vectors
        :param state_size: The size of the internal state of the transformer
        '''
        super().__init__()

        assert scale_mode in ["none", "opennmt", "down"]
        assert embedding_init in ["pytorch", "xavier", "kaiming"]

        assert (not same_enc_dec_embedding) or (n_input_tokens == n_out_tokens)

        self.tied_embedding = tied_embedding

        self.decoder_sos_eos = n_out_tokens
        self.encoder_eos = n_input_tokens
        self.encoder_sos = n_input_tokens + 1 if encoder_sos else None
        self.state_size = state_size
        self.embedding_init = embedding_init
        self.ff_multiplier = ff_multiplier
        self.n_input_tokens = n_input_tokens
        self.n_out_tokens = n_out_tokens
        self.in_embedding_size = in_embedding_size
        self.out_embedding_size = out_embedding_size
        self.same_enc_dec_embedding = same_enc_dec_embedding
        self.scale_mode = scale_mode
        self.pos = pos_embeddig or framework.layers.PositionalEncoding(state_size, max_len=max_len, batch_first=True,
                                        scale=(1.0 / math.sqrt(state_size)) if scale_mode == "down" else 1.0)
        
        self.register_buffer('int_seq', torch.arange(max_len, dtype=torch.long))
        self.construct(transformer, **kwargs)
        self.reset_parameters()

    def pos_embed(self, t: torch.Tensor, offset: int, scale_offset: int) -> torch.Tensor:
        if self.scale_mode == "opennmt":
            t = t * math.sqrt(t.shape[-1])
        
        return self.pos(t, offset)

    def construct(self, transformer, **kwargs):
        self.input_embedding = torch.nn.Embedding(self.n_input_tokens + 1 + int(self.encoder_sos is not None), 
                                                  self.in_embedding_size or self.state_size)
        self.output_embedding = self.input_embedding if self.same_enc_dec_embedding else \
                                torch.nn.Embedding(self.n_out_tokens+1, self.out_embedding_size or self.state_size)

        if self.in_embedding_size is not None:
            self.in_embedding_upscale = torch.nn.Linear(self.in_embedding_size, self.state_size)
        
        if self.out_embedding_size is not None:
            self.out_embedding_upscale = torch.nn.Linear(self.out_embedding_size, self.state_size)

        if self.tied_embedding:
            assert self.out_embedding_size is None
            self.output_map = TiedEmbedding(self.output_embedding.weight)
        else:
            self.output_map = torch.nn.Linear(self.state_size, self.n_out_tokens+1)

        self.trafo = transformer(d_model=self.state_size, dim_feedforward=int(self.ff_multiplier*self.state_size),
                                 **kwargs)

    def reset_parameters(self):
        if self.embedding_init == "xavier":
            torch.nn.init.xavier_uniform_(self.input_embedding.weight)
            torch.nn.init.xavier_uniform_(self.output_embedding.weight)
        elif self.embedding_init == "kaiming":
            torch.nn.init.kaiming_normal_(self.input_embedding.weight)
            torch.nn.init.kaiming_normal_(self.output_embedding.weight)

        if not self.tied_embedding:
            torch.nn.init.xavier_uniform_(self.output_map.weight)

    def generate_len_mask(self, max_len: int, len: torch.Tensor) -> torch.Tensor:
        return self.int_seq[: max_len] >= len.unsqueeze(-1)

    def output_embed(self, x: torch.Tensor) -> torch.Tensor:
        o = self.output_embedding(x)
        if self.out_embedding_size is not None:
            o = self.out_embedding_upscale(o)
        return o

    def run_greedy(self, src: torch.Tensor, src_len: torch.Tensor, max_len: int) -> TransformerResult:
        batch_size = src.shape[0]
        n_steps = src.shape[1]

        in_len_mask = self.generate_len_mask(n_steps, src_len)
        memory = self.trafo.encoder(src, mask=in_len_mask)

        running = torch.ones([batch_size], dtype=torch.bool, device=src.device)
        out_len = torch.zeros_like(running, dtype=torch.long)

        next_tgt = self.pos_embed(self.output_embed(torch.full([batch_size,1], self.decoder_sos_eos, dtype=torch.long,
                                                                device=src.device)), 0, 1)

        all_outputs = []
        state = self.trafo.decoder.create_state(src.shape[0], max_len, src.device)

        for i in range(max_len):
            output = self.trafo.decoder.one_step_forward(state, next_tgt, memory, memory_key_padding_mask=in_len_mask)

            output = self.output_map(output)
            all_outputs.append(output)

            out_token = torch.argmax(output[:,-1], -1)
            running &= out_token != self.decoder_sos_eos

            out_len[running] = i + 1
            next_tgt = self.pos_embed(self.output_embed(out_token).unsqueeze(1), i+1, 1)

        return TransformerResult.create(torch.cat(all_outputs, 1), out_len)

    def run_teacher_forcing(self, src: torch.Tensor, src_len: torch.Tensor, target: torch.Tensor,
                            target_len: torch.Tensor) -> TransformerResult:
        target = self.output_embed(F.pad(target[:, :-1], (1, 0), value=self.decoder_sos_eos).long())
        target = self.pos_embed(target, 0, 1)

        in_len_mask = self.generate_len_mask(src.shape[1], src_len)

        res = self.trafo(src, target, src_length_mask=in_len_mask,
                          tgt_mask=self.trafo.generate_square_subsequent_mask(target.shape[1], src.device))

        return TransformerResult.create(self.output_map(res), target_len)

    def input_embed(self, x: torch.Tensor) -> torch.Tensor:
        src = self.input_embedding(x.long())
        if self.in_embedding_size is not None:
            src = self.in_embedding_upscale(src)
        return src

    def forward(self, src: torch.Tensor, src_len: torch.Tensor, target: torch.Tensor,
                target_len: torch.Tensor, teacher_forcing: bool, max_len: Optional[int] = None) -> TransformerResult:
        '''
        Run transformer encoder-decoder on some input/output pair

        :param src: source tensor. Shape: [N, S], where S in the in sequence length, N is the batch size
        :param src_len: length of source sequences. Shape: [N], N is the batch size
        :param target: target tensor. Shape: [N, S], where T in the in sequence length, N is the batch size
        :param target_len: length of target sequences. Shape: [N], N is the batch size
        :param teacher_forcing: use teacher forcing or greedy decoding
        :param max_len: overwrite autodetected max length. Useful for parallel execution
        :return: prediction of the target tensor. Shape [N, T, C_out]
        '''

        if self.encoder_sos is not None:
            src = F.pad(src, (1, 0), value=self.encoder_sos)
            src_len = src_len + 1
            
        src = self.pos_embed(self.input_embed(src), 0, 0)

        if teacher_forcing:
            return self.run_teacher_forcing(src, src_len, target, target_len)
        else:
            return self.run_greedy(src, src_len, max_len or target_len.max().item())
