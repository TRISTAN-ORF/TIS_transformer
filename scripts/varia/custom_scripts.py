
import torch
from torch import nn, einsum
import torch.nn.functional as F
import math
from functools import partial
from einops import rearrange, pack
from performer_pytorch.performer_pytorch import PreLayerNorm, FeedForward, PreScaleNorm, ReZero, cast_tuple, \
    Chunk, PreShiftTokens, CrossAttention, ReversibleSequence, SequentialSequence, ProjectionUpdater, \
        default, FastAttention, exists, rearrange, empty, apply_rotary_pos_emb

def compl_mod(m, n):
    return int(n * math.ceil(m/n) - m)

class SinusoidalEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x):
        n = x.shape[-2]
        t = torch.arange(n, device = x.device).type_as(self.inv_freq)
        freqs = torch.einsum('i , j -> i j', t, self.inv_freq)
        return torch.cat((freqs, freqs), dim=-1)

def rotate_half(x):
    x = rearrange(x, 'b ... (r d) -> b (...) r d', r = 2)
    x1, x2 = x.unbind(dim = -2)
    return torch.cat((-x2, x1), dim = -1)

def apply_rotary_pos_emb(q, k, freqs):
    q, k = map(lambda t: (t * freqs.cos()) + (rotate_half(t) * freqs.sin()), (q, k))
    return q, k

class WindowAttention(nn.Module):
    def __init__(self, window, dim, dropout=0.1):
        super().__init__()
        assert window % 2 == 1, 'Window size should be an odd integer.'
        
        self.softmax = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)
        self.w = int((window-1)/2)
        
        self.k_ch = window * 2
        self.q_ch = window + 1
        
        u = torch.triu(torch.full((self.q_ch, self.k_ch), True))
        self.mask = ~torch.logical_and(u, torch.flip(u,[0,1]))
        self.mask_k_left = torch.clone(self.mask)
        self.mask_k_left[:,:self.w] = True
        self.rel_pos = SinusoidalEmbeddings(dim)
        
    
    def forward(self, q, k, v, input_mask):
        assert k.shape[2] == q.shape[2], 'q and k should have same input length.'
        b, nh, s, h = k.shape
        
        q = q * (h ** -.5)
        
        (q, packed_shape), (k, _), (v, _) = map(lambda t: pack([t], '* n d'), (q, k, v))
        
        pos_emb = self.rel_pos(q)
        q, k = apply_rotary_pos_emb(q, k, pos_emb)
        
        if s < self.w*2.5:
            A = torch.einsum('b q h, b k h -> b q k', q, k)
            u = torch.triu(torch.full(A.shape[-2:], True), -self.w).to(A.device)
            mask = torch.logical_and(u, torch.flip(u,[0,1]))
            A[:].masked_fill_(~mask, -torch.finfo(A.dtype).max)
            A = self.softmax(A)
            A = self.dropout(A)
            
            z = torch.einsum('b q k, b k h -> b q h', A, v)
            z = z.view(b, nh, -1, h)
        else:
            q_pad = compl_mod(s, self.q_ch)
            k_pad = compl_mod((s + self.w*2)-self.k_ch, self.q_ch)
        
            q = F.pad(q, (0,)*3 + (q_pad,)).unfold(1, self.q_ch, self.q_ch)
            k = F.pad(k, (0,)*2 + (self.w, self.w + k_pad)).unfold(1, self.k_ch, self.q_ch)
            v = F.pad(v, (0,)*2 + (self.w, self.w + k_pad)).unfold(1, self.k_ch, self.q_ch)
        
            A = einsum('b c h q, b c h k -> b c q k ', q, k)
        
            mask_value = -torch.finfo(A.dtype).max
            mask_k_right = torch.clone(self.mask.to(A.device))
            mask_k_right[:,-(self.w+k_pad):] = True
            if q.shape[1] > 1:
                mask = torch.stack([self.mask_k_left.to(A.device)] + \
                                [self.mask.to(A.device)]*(q.shape[1]-2) + \
                                [mask_k_right])
            else:
                mask = torch.logical_or(self.mask_k_left.to(A.device), mask_k_right)
        
            A[:].masked_fill_(mask, mask_value)
            A = self.softmax(A)
            A = self.dropout(A)
        
            z = einsum('b c q k, b c h k -> b c q h', A, v)
            z = z.view(b, nh, -1, h)[:,:,:s]
        
        return z
    
# Code adapted from https://github.com/lucidrains/performer-pytorch
class Performer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        dim_head,
        local_attn_heads = 0,
        local_window_size = 256,
        causal = False,
        ff_mult = 4,
        nb_features = None,
        feature_redraw_interval = 1000,
        reversible = False,
        ff_chunks = 1,
        generalized_attention = False,
        kernel_fn = nn.ReLU(),
        use_scalenorm = False,
        use_rezero = False,
        ff_glu = False,
        ff_dropout = 0.,
        attn_dropout = 0.,
        cross_attend = False,
        no_projection = False,
        auto_check_redraw = True,
        qkv_bias = True,
        attn_out_bias = True,
        shift_tokens = False
    ):
        super().__init__()
        layers = nn.ModuleList([])
        local_attn_heads = cast_tuple(local_attn_heads)
        local_attn_heads = local_attn_heads * depth if len(local_attn_heads) == 1 else local_attn_heads
        assert len(local_attn_heads) == depth, 'tuple specifying number of local attention heads per depth must be equal to the total depth'
        assert all(map(lambda n: n >= 0 and n <= heads, local_attn_heads)), 'local attention head value must be less than the total number of heads'

        if use_scalenorm:
            wrapper_fn = partial(PreScaleNorm, dim)
        elif use_rezero:
            wrapper_fn = ReZero
        else:
            wrapper_fn = partial(PreLayerNorm, dim)

        for _, local_heads in zip(range(depth), local_attn_heads):

            attn = SelfAttention(dim, causal = causal, heads = heads, dim_head = dim_head, local_heads = local_heads, local_window_size = local_window_size, nb_features = nb_features, generalized_attention = generalized_attention, kernel_fn = kernel_fn, dropout = attn_dropout, no_projection = no_projection, qkv_bias = qkv_bias, attn_out_bias = attn_out_bias)
            ff = Chunk(ff_chunks, FeedForward(dim, mult = ff_mult, dropout = ff_dropout, glu = ff_glu), along_dim = 1)

            if shift_tokens:
                shift = (0, 1) if causal else (-1, 0, 1)
                attn, ff = map(lambda t: PreShiftTokens(shift, t), (attn, ff))

            attn, ff = map(wrapper_fn, (attn, ff))
            layers.append(nn.ModuleList([attn, ff]))

            if not cross_attend:
                continue

            layers.append(nn.ModuleList([
                wrapper_fn(CrossAttention(dim, heads = heads, dim_head = dim_head, nb_features = nb_features, generalized_attention = generalized_attention, kernel_fn = kernel_fn, dropout = attn_dropout, no_projection = no_projection, qkv_bias = qkv_bias, attn_out_bias = attn_out_bias)),
                wrapper_fn(Chunk(ff_chunks, FeedForward(dim, mult = ff_mult, dropout = ff_dropout, glu = ff_glu), along_dim = 1))
            ]))

        execute_type = ReversibleSequence if reversible else SequentialSequence

        route_attn = ((True, False),) * depth * (2 if cross_attend else 1)
        route_context = ((False, False), (True, False)) * depth
        attn_route_map = {'mask': route_attn, 'pos_emb': route_attn}
        context_route_map = {'context': route_context, 'context_mask': route_context} if cross_attend else {}
        self.net = execute_type(layers, args_route = {**attn_route_map, **context_route_map})

        # keeping track of when to redraw projections for all attention layers
        self.auto_check_redraw = auto_check_redraw
        self.proj_updater = ProjectionUpdater(self.net, feature_redraw_interval)

    def fix_projection_matrices_(self):
        self.proj_updater.feature_redraw_interval = None

    def forward(self, x, **kwargs):
        if self.auto_check_redraw:
            self.proj_updater.redraw_projections()
        return self.net(x, **kwargs)
        
class Attention(nn.Module):
    def __init__(
        self,
        dim,
        causal = False,
        heads = 8,
        dim_head = 64,
        local_heads = 0,
        local_window_size = 256,
        nb_features = None,
        feature_redraw_interval = 1000,
        generalized_attention = False,
        kernel_fn = nn.ReLU(),
        dropout = 0.,
        no_projection = False,
        qkv_bias = False,
        attn_out_bias = True
    ):
        super().__init__()
        assert dim % heads == 0, 'dimension must be divisible by number of heads'
        dim_head = default(dim_head, dim // heads)
        inner_dim = dim_head * heads
        self.fast_attention = FastAttention(dim_head, nb_features, causal = causal, generalized_attention = generalized_attention, kernel_fn = kernel_fn, no_projection = no_projection)

        self.heads = heads
        self.global_heads = heads - local_heads
        #self.local_attn = LocalAttention(window_size = local_window_size, causal = causal, autopad = True, dropout = dropout, look_forward = int(not causal), rel_pos_emb_config = (dim_head, local_heads)) if local_heads > 0 else None
        self.local_attn = WindowAttention(window=local_window_size, dim=dim_head, dropout=0.10)

        self.to_q = nn.Linear(dim, inner_dim, bias = qkv_bias)
        self.to_k = nn.Linear(dim, inner_dim, bias = qkv_bias)
        self.to_v = nn.Linear(dim, inner_dim, bias = qkv_bias)
        self.to_out = nn.Linear(inner_dim, dim, bias = attn_out_bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, pos_emb = None, context = None, mask = None, context_mask = None, **kwargs):
        b, n, _, h, gh = *x.shape, self.heads, self.global_heads

        cross_attend = exists(context)

        context = default(context, x)
        context_mask = default(context_mask, mask) if not cross_attend else context_mask

        q, k, v = self.to_q(x), self.to_k(context), self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))
        (q, lq), (k, lk), (v, lv) = map(lambda t: (t[:, :gh], t[:, gh:]), (q, k, v))

        attn_outs = []

        if not empty(q):
            if exists(context_mask):
                global_mask = context_mask[:, None, :, None]
                v.masked_fill_(~global_mask, 0.)

            if exists(pos_emb) and not cross_attend:
                q, k = apply_rotary_pos_emb(q, k, pos_emb)

            out = self.fast_attention(q, k, v)
            attn_outs.append(out)
        # local attention
        if not empty(lq):
            assert not cross_attend, 'local attention is not compatible with cross attention'
            # lq=lk=lv :B x heads x L x dim_head
            #out = self.local_attn(lq.permute(0,2,1,3), lk.permute(0,2,1,3), lv.permute(0,2,1,3), input_mask = mask).permute(0,2,1,3)
            out = self.local_attn(lq, lk, lv, input_mask = mask)
            attn_outs.append(out)

        out = torch.cat(attn_outs, dim = 1)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return self.dropout(out)
    
class SelfAttention(Attention):
    def forward(self, *args, context = None, **kwargs):
        assert not exists(context), 'self attention should not receive context'
        return super().forward(*args, **kwargs)