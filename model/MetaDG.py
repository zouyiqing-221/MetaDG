import numpy as np
import torch
import torch.nn as nn

def asym_adj(adj):
    row_sum = adj.sum(-1)
    d_inv = 1 / row_sum
    if adj.dim() == 2:
        adj = torch.einsum('nm,n->nm', adj, d_inv)
    elif adj.dim() == 3:
        adj = torch.einsum('bnm,bn->bnm', adj, d_inv)
    return adj

class InstanceNorm(nn.Module):
    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=[-2, -1], keepdim=True)
        var = x.var(dim=[-2, -1], unbiased=False, keepdim=True)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta

class GCN(nn.Module):
    def __init__(self):
        super(GCN, self).__init__()

    def forward(self, x, adj):
        if adj.dim() == 2:
            x = torch.einsum('nm,bmd->bnd', adj, x)
        elif adj.dim() == 3:
            x = torch.einsum('bnm,bmd->bnd', adj, x)
        else:
            raise Exception(f'Wrong Adj Matrix Dimension! Assert 2 or 3, got {adj.dim()}.')
        return x.contiguous()

class VariationalDropout(nn.Module):
    def __init__(self, p, device):
        super(VariationalDropout, self).__init__()
        self.dropout_rate = p
        self.device = device
        self.mask = None

    def forward(self, x, time_step):
        if self.training:
            if time_step == 0:
                self.mask = self.generate_mask(x.shape)
            x = self.mask * x
        return x

    def generate_mask(self, tensor_shape, device="cuda"):
        """
        生成变分Dropout掩码（同一序列的所有时间步复用相同掩码）
        Args:
            tensor_shape: 输入张量的形状，假设为 (batch_size, hidden_size) 或 (seq_len, batch_size, hidden_size)
            device: 掩码所在的设备
        Returns:
            mask: 形状为 (batch_size, hidden_size) 的掩码张量
        """
        if len(tensor_shape) == 3:
            # 输入是三维张量 (batch_size, node_num, hidden_size)
            batch_size, node_num, hidden_size = tensor_shape[0], tensor_shape[1], tensor_shape[2]
        else:
            # 输入是二维张量 (batch_size, hidden_size)
            batch_size, hidden_size = tensor_shape[0], tensor_shape[1]

        # 生成掩码：每个样本独立，每个特征维度随机丢弃
        mask = torch.bernoulli(torch.ones([batch_size, node_num, hidden_size], device=self.device) * (1 - self.dropout_rate))
        mask = mask / (1 - self.dropout_rate)  # 缩放激活值
        return mask  # shape: (batch_size, hidden_size)

class MetaDGCN(nn.Module):
    def __init__(self, hop_num_k, hidden_dim, out_dim, graph_num, node_emb_dim):
        super(MetaDGCN, self).__init__()

        self.hop_num_k = hop_num_k
        self.hidden_dim = hidden_dim
        self.graph_num = graph_num
        self.node_emb_dim = node_emb_dim

        self.gconv = GCN()

        self.weights_pool = nn.init.xavier_normal_(nn.Parameter(torch.empty([node_emb_dim, (graph_num * hop_num_k + 1) * hidden_dim, out_dim])))
        self.bias_pool = nn.init.xavier_normal_(nn.Parameter(torch.empty([node_emb_dim, out_dim])))

    def forward(self, x, graph_list, node_emb):
        out = [x]
        for i in range(self.graph_num):
            x1 = x
            for k in range(self.hop_num_k):
                x2 = self.gconv(x1, graph_list[i])
                out.append(x2)
                x1 = x2

        h = torch.cat(out, dim=-1)
        weights = torch.einsum('bnd,dio->bnio', node_emb, self.weights_pool)
        bias = torch.einsum('bnd,do->bno', node_emb, self.bias_pool)
        h = torch.einsum('bni,bnio->bno', h, weights) + bias

        return h

class DynamicGraphQualification(nn.Module):
    def __init__(self, node_emb_dim, delta=2., device=None):
        super(DynamicGraphQualification, self).__init__()

        self.delta = delta
        self.threshold_pool = nn.init.xavier_normal_(nn.Parameter(torch.FloatTensor(node_emb_dim, 1)))
        self.norm_phi = InstanceNorm()
        self.device = device

    def qualify(self, current_node_emb=None, prev_node_emb=None, static_node_emb=None):
        # static_node_emb.dim() == 2
        b, n, _ = current_node_emb.size()
        graph = torch.einsum('bnd,bmd->bnm', current_node_emb, prev_node_emb)
        static_graph_mask = torch.greater(torch.relu(torch.einsum('nd,md->nm', static_node_emb, static_node_emb)),
                                          torch.zeros([n, n]).to(self.device)).float()
        probs = asym_adj(torch.relu(torch.einsum('bnm,nm->bnm', graph, static_graph_mask)) + 1e-10)

        threshold = torch.sigmoid(torch.einsum('bnd,dg->bng', current_node_emb, self.threshold_pool))
        threshold_baseline = torch.diagonal(probs, dim1=-2, dim2=-1).unsqueeze(-1)
        threshold = threshold_baseline * threshold
        x = probs - threshold
        pos_mask = torch.greater_equal(x, torch.zeros([b, n, 1]).to(self.device)).float()
        neg_mask = 1 - pos_mask

        pos_mask = torch.sigmoid(x) * pos_mask
        tau_beta = torch.exp(self.norm_phi(pos_mask) * self.delta)
        phi = tau_beta * pos_mask + tau_beta * neg_mask

        return phi

    def forward(self, static_node_emb, current_node_emb, prev_node_emb):
        """

        :param static_node_emb: N
        :param current_node_emb: N_t^m
        :param prev_node_emb: N_{t-1}^m
        :return: phi_t
        """
        current_phi = self.qualify(current_node_emb=current_node_emb, prev_node_emb=prev_node_emb, static_node_emb=static_node_emb)
        return current_phi

class SpatialTemporalCorrelationEnhancement(nn.Module):
    def __init__(self, node_emb_dim, hidden_emb_dim, cross_attn_hidden_dim, corr_enhance_mode, device):
        super(SpatialTemporalCorrelationEnhancement, self).__init__()

        self.corr_enhance_mode = corr_enhance_mode  # ['S', 'T', 'ST', 'TS', 'OFF']

        if 'S' in self.corr_enhance_mode:
            self.linear_q = nn.Linear(node_emb_dim, cross_attn_hidden_dim)
            self.linear_k = nn.Linear(node_emb_dim, cross_attn_hidden_dim)
            self.linear_v = nn.Linear(node_emb_dim, cross_attn_hidden_dim)
            self.linear_out = nn.Linear(cross_attn_hidden_dim, node_emb_dim)
            self.linear_out2 = nn.Linear(node_emb_dim, node_emb_dim)
            self.dropout1 = VariationalDropout(p=0.1, device=device)
            self.dropout2 = VariationalDropout(p=0.1, device=device)

        if 'T' in self.corr_enhance_mode:
            # print(f'STRUCT-DYNAMIC-PARAMS-GRAPHS-BY-RESET')
            self.reset_proj = nn.Linear(hidden_emb_dim, node_emb_dim)

        self.scale = cross_attn_hidden_dim ** -0.5

    def spatial_correlation_enhancement(self, current_dynamic_node_emb, prev_dynamic_node_emb, time_step):
        q = self.linear_q(current_dynamic_node_emb)
        k = self.linear_k(prev_dynamic_node_emb)
        v = self.linear_v(prev_dynamic_node_emb)

        cross_attn = torch.einsum('bnh,bmh->bnm', q, k) * self.scale
        norm_cross_attn = torch.softmax(cross_attn, dim=-1)

        current_sce_node_emb = torch.einsum('bnm,bmh->bnh', norm_cross_attn, v)

        # (ln,drop)
        current_sce_node_emb = self.dropout1(current_dynamic_node_emb + self.linear_out(current_sce_node_emb), time_step)
        current_sce_node_emb = self.dropout2(current_dynamic_node_emb + self.linear_out2(current_sce_node_emb), time_step)

        return current_sce_node_emb

    def temporal_correlation_enhancement(self, current_dynamic_node_emb, prev_dynamic_node_emb, z):
        z_hat = torch.sigmoid(self.reset_proj(z))
        current_tce_node_emb = z_hat * prev_dynamic_node_emb + (1 - z_hat) * current_dynamic_node_emb
        return current_tce_node_emb

    def forward(self, current_dynamic_node_emb, prev_dynamic_node_emb, prev_enhanced_node_emb, z=None, time_step=None):
        """

        :param current_dynamic_node_emb: N_t
        :param prev_dynamic_node_emb: N_{t-1}
        :param prev_enhanced_node_emb: N_{t-1}^*
        :return: N_t^*
        """
        if self.corr_enhance_mode == 'ST':
            # STCE
            current_enhanced_node_emb = self.spatial_correlation_enhancement(current_dynamic_node_emb, prev_dynamic_node_emb, time_step)
            if z is not None:
                current_enhanced_node_emb = self.temporal_correlation_enhancement(current_enhanced_node_emb, prev_enhanced_node_emb, z)
        elif self.corr_enhance_mode == 'TS':
            # TSCE
            if z is not None:
                current_enhanced_node_emb = self.temporal_correlation_enhancement(current_dynamic_node_emb, prev_dynamic_node_emb, z)
            current_enhanced_node_emb = self.spatial_correlation_enhancement(current_enhanced_node_emb, prev_enhanced_node_emb, time_step)
        elif self.corr_enhance_mode == 'S':
            # SCE
            current_enhanced_node_emb = self.spatial_correlation_enhancement(current_dynamic_node_emb, prev_dynamic_node_emb, time_step)
        elif self.corr_enhance_mode == 'T':
            # TCE
            if z is not None:
                current_enhanced_node_emb = self.temporal_correlation_enhancement(current_dynamic_node_emb, prev_dynamic_node_emb, z)

        return current_enhanced_node_emb

class DynamicNodeGeneration(nn.Module):
    def __init__(self, node_emb_dim, time_emb_dim, hidden_emb_dim, graph_num):
        super(DynamicNodeGeneration, self).__init__()

        self.graph_num = graph_num
        gamma_src_dim = time_emb_dim * 2
        self.state_proj = nn.Linear(hidden_emb_dim, node_emb_dim)
        self.gamma_pool = nn.init.xavier_normal_(nn.Parameter(torch.FloatTensor(gamma_src_dim, graph_num, node_emb_dim)))

    def get_gamma(self, gamma_pool, emb=None):
        if emb.dim() == 2:
            gamma = torch.einsum('bd,dh->bh', emb, gamma_pool).unsqueeze(1)
        elif emb.dim() == 3:
            gamma = torch.einsum('bnd,dh->bnh', emb, gamma_pool)
        gamma = torch.sigmoid(gamma)
        return gamma

    def forward(self, static_node_emb, time_emb, state):
        """

        :param static_node_emb: N
        :param time_emb: T_t
        :param state: H_{t-1}
        :return: N_t
        """
        state_emb = self.state_proj(state)
        # B, N, D = state_emb.size()
        current_dynamic_node_emb = []
        for i in range(self.graph_num):
            src_emb = torch.cat([time_emb[0], time_emb[1]], dim=-1)
            gamma = self.get_gamma(self.gamma_pool[:, i, :], src_emb)
            current_reserve_emb = gamma * static_node_emb + (1 - gamma) * state_emb
            # current_emb = current_reserve_emb
            current_dynamic_node_emb.append(current_reserve_emb)
        current_dynamic_node_emb = torch.stack(current_dynamic_node_emb, dim=1).squeeze(dim=1)

        return current_dynamic_node_emb

class MetaDGCRU(nn.Module):
    def __init__(self,
                 hop_num_k, in_dim, hidden_dim, out_dim, node_emb_dim, time_emb_dim, continuous_time_emb_dim, node_num, graph_num,
                 use_mask, mask_pool, refine_dynamic_node=True, refine_dynamic_graph=True, corr_enhance_mode='ST', device=None):
        super(MetaDGCRU, self).__init__()

        self.node_num = node_num
        self.hidden_dim = hidden_dim
        self.refine_dynamic_node = refine_dynamic_node
        self.refine_dynamic_graph = refine_dynamic_graph

        self.dynamic_node_generation = DynamicNodeGeneration(node_emb_dim=node_emb_dim,
                                                             time_emb_dim=time_emb_dim,
                                                             hidden_emb_dim=hidden_dim,
                                                             graph_num=graph_num)

        if self.refine_dynamic_node and self.refine_dynamic_graph:    # MetaDG
            self.stce_p = SpatialTemporalCorrelationEnhancement(node_emb_dim=node_emb_dim,
                                                                hidden_emb_dim=hidden_dim,
                                                                cross_attn_hidden_dim=64,
                                                                corr_enhance_mode=corr_enhance_mode,
                                                                device=device)
            self.phi = DynamicGraphQualification(node_emb_dim=node_emb_dim, delta=2., device=device)
            self.stce_g = SpatialTemporalCorrelationEnhancement(node_emb_dim=node_emb_dim,
                                                                hidden_emb_dim=hidden_dim,
                                                                cross_attn_hidden_dim=64,
                                                                corr_enhance_mode=corr_enhance_mode,
                                                                device=device)
            self.stce_m = SpatialTemporalCorrelationEnhancement(node_emb_dim=node_emb_dim,
                                                                hidden_emb_dim=hidden_dim,
                                                                cross_attn_hidden_dim=64,
                                                                corr_enhance_mode=corr_enhance_mode,
                                                                device=device)
        elif self.refine_dynamic_node:   # w/o DGQ
            self.stce_p = SpatialTemporalCorrelationEnhancement(node_emb_dim=node_emb_dim,
                                                                hidden_emb_dim=hidden_dim,
                                                                cross_attn_hidden_dim=64,
                                                                corr_enhance_mode=corr_enhance_mode,
                                                                device=device)
            self.stce_g = SpatialTemporalCorrelationEnhancement(node_emb_dim=node_emb_dim,
                                                                hidden_emb_dim=hidden_dim,
                                                                cross_attn_hidden_dim=64,
                                                                corr_enhance_mode=corr_enhance_mode,
                                                                device=device)
        elif self.refine_dynamic_graph:  # w/o STCE
            self.phi = DynamicGraphQualification(node_emb_dim=node_emb_dim, delta=2., device=device)

        self.use_mask = use_mask
        self.mask_pool = mask_pool
        if self.use_mask:
            p = 0.3
            print(f'MASK DROPOUT {p}')
            self.dropout = nn.Dropout(p=p)
            # self.dropout = VariationalDropout(p=p, device=device)
            if mask_pool is None:
                mask_emb_dim = node_emb_dim * 4
                self.mask_pool = nn.init.xavier_normal_(nn.Parameter(torch.FloatTensor(continuous_time_emb_dim * 2, 1, node_num, mask_emb_dim)))

        gcn_in_dim = in_dim + hidden_dim
        self.gate = MetaDGCN(hop_num_k, gcn_in_dim, out_dim * 2, graph_num, node_emb_dim)
        self.update = MetaDGCN(hop_num_k, gcn_in_dim, out_dim, graph_num, node_emb_dim)
        self.device = device

    def get_mask(self, time_emb, node_emb, mask_pool):
        mask_pool = torch.einsum('bns,dsh->bdnh', node_emb, mask_pool)
        mask_emb = torch.einsum('bd,bdnh->bnh', time_emb, mask_pool)
        graph_mask = torch.einsum('bnh,bmh->bnm', mask_emb, mask_emb)
        graph_mask = torch.relu(graph_mask)
        return graph_mask

    def forward(self,
                x, state, graph_list, static_node_emb, time_emb, continuous_time_emb,
                prev_dynamic_node_emb, prev_stce_p, prev_stce_g, prev_stce_m, z, time_step):
        current_dynamic_node_emb = self.dynamic_node_generation(static_node_emb, time_emb, state)
        if self.refine_dynamic_node and self.refine_dynamic_graph:
            current_graph_list = []
            for i, graph in enumerate(graph_list):
                B, _, _ = current_dynamic_node_emb.size()
                if time_step == 0:
                    prev_dynamic_node_emb = prev_dynamic_node_emb.repeat(B, 1, 1)
                    prev_stce_p = prev_stce_p.repeat(B, 1, 1)
                    prev_stce_g = prev_stce_g.repeat(B, 1, 1)
                    prev_stce_m = prev_stce_m.repeat(B, 1, 1)

                current_stce_p = self.stce_p(current_dynamic_node_emb, prev_dynamic_node_emb, prev_stce_p, z=z, time_step=time_step)

                current_stce_g = self.stce_g(current_dynamic_node_emb, prev_dynamic_node_emb, prev_stce_g, z=z, time_step=time_step)
                graph = torch.einsum('bnd,bmd->bnm', current_stce_g, current_stce_g)
                if self.use_mask:
                    graph_mask = self.get_mask(continuous_time_emb, current_stce_g, self.mask_pool[:, i, :, :])
                    graph = graph_mask * graph

                current_stce_m = self.stce_m(current_dynamic_node_emb, prev_dynamic_node_emb, prev_stce_m, z=z, time_step=time_step)
                current_phi = self.phi(static_node_emb, current_stce_m, prev_stce_m)
                graph = torch.einsum('bnm,bnm->bnm', current_phi, graph)

                # Row Normalization
                if self.use_mask:
                    graph = asym_adj(self.dropout(torch.relu(graph)) + 1e-10)
                else:
                    graph = asym_adj(torch.relu(graph) + 1e-10)

                current_graph_list.append(graph)
        elif self.refine_dynamic_node:
            current_stce_m = None
            current_graph_list = []
            for i, graph in enumerate(graph_list):
                current_stce_p = self.stce_p(current_dynamic_node_emb, prev_dynamic_node_emb, prev_stce_p, z=z, time_step=time_step)

                current_stce_g = self.stce_g(current_dynamic_node_emb, prev_dynamic_node_emb, prev_stce_g, z=z, time_step=time_step)
                graph = torch.einsum('bnd,bmd->bnm', current_stce_g, current_stce_g)
                if self.use_mask:
                    graph_mask = self.get_mask(continuous_time_emb, current_stce_g, self.mask_pool[:, i, :, :])
                    graph = graph_mask * graph
                    graph = asym_adj(self.dropout(torch.relu(graph)) + 1e-10)
                else:
                    graph = asym_adj(torch.relu(graph) + 1e-10)
                current_graph_list.append(graph)
        elif self.refine_dynamic_graph:
            current_stce_p, current_stce_g, current_stce_m = None, None, None
            current_graph_list = []
            for i, graph in enumerate(graph_list):
                graph = torch.einsum('bnd,bmd->bnm', current_dynamic_node_emb, current_dynamic_node_emb)
                if self.use_mask:
                    graph_mask = self.get_mask(continuous_time_emb, current_dynamic_node_emb, self.mask_pool[:, i, :, :])
                    graph = graph_mask * graph
                current_phi = self.phi(static_node_emb, current_dynamic_node_emb, prev_dynamic_node_emb)
                graph = torch.einsum('bnm,bnm->bnm', current_phi, graph)
                if self.use_mask:
                    graph = asym_adj(self.dropout(torch.relu(graph)) + 1e-10)
                else:
                    graph = asym_adj(torch.relu(graph) + 1e-10)
                current_graph_list.append(graph)

        input_and_state = torch.cat((x, state), dim=-1)
        r_z = torch.sigmoid(self.gate(input_and_state, current_graph_list, current_stce_p))
        r, z = torch.split(r_z, self.hidden_dim, dim=-1)
        temp_state = r * state
        temp = torch.cat((x, temp_state), dim=-1)
        c = torch.tanh(self.update(temp, current_graph_list, current_stce_p))
        h = z * state + (1 - z) * c
        return h, z, current_dynamic_node_emb, current_stce_p, current_stce_g, current_stce_m

    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.node_num, self.hidden_dim)

class Encoder(nn.Module):
    def __init__(self,
                 hop_num_k, in_dim, hidden_dim, out_dim, node_emb_dim, time_emb_dim, continuous_time_emb_dim,
                 node_num, graph_num, layer_num, use_mask, mask_pool=None, refine_dynamic_node=True, refine_dynamic_graph=True, corr_enhance_mode='ST',
                 device=None):
        super(Encoder, self).__init__()

        self.cells = nn.ModuleList()
        for i in range(layer_num):
            self.cells.append(MetaDGCRU(hop_num_k, in_dim, hidden_dim, out_dim, node_emb_dim, time_emb_dim, continuous_time_emb_dim,
                                        node_num, graph_num, use_mask, mask_pool,
                                        refine_dynamic_node=refine_dynamic_node, refine_dynamic_graph=refine_dynamic_graph, corr_enhance_mode=corr_enhance_mode,
                                        device=device))
        self.device = device

    def forward(self, x, graph_list, node_emb=None, time_emb=None, continuous_time_emb=None, use_mask=True):
        b, steps, n, _ = x.size()
        output_hidden = []
        for cell in self.cells:
            prev_dynamic_node_emb = prev_stce_p = prev_stce_g = prev_stce_m = node_emb
            z = None
            state = cell.init_hidden_state(b)
            inner_states = []
            for t in range(steps):
                if use_mask:
                    continuous_time_emb_t = continuous_time_emb[:, t, ...]
                else:
                    continuous_time_emb_t = None
                state, z, prev_dynamic_node_emb, prev_stce_p, prev_stce_g, prev_stce_m = cell(x[:, t, :, :], state, graph_list, node_emb,
                                                                                                [time_emb[:, 0, :], time_emb[:, t, :]],
                                                                                                continuous_time_emb_t,
                                                                                                prev_dynamic_node_emb,
                                                                                                prev_stce_p, prev_stce_g, prev_stce_m,
                                                                                                z=z, time_step=t)
                inner_states.append(state)
            output_hidden.append(state)
            current_input = torch.stack(inner_states, dim=1)
        return current_input, output_hidden

class Decoder(nn.Module):
    def __init__(self,
                 hop_num_k, in_dim, hidden_dim, out_dim, node_emb_dim, time_emb_dim, continuous_time_emb_dim,
                 node_num, graph_num, layer_num, use_mask, mask_pool=None, refine_dynamic_node=True, refine_dynamic_graph=True, corr_enhance_mode='ST',
                 device=None):
        super(Decoder, self).__init__()

        self.cells = nn.ModuleList()
        for i in range(layer_num):
            self.cells.append(MetaDGCRU(hop_num_k, in_dim, hidden_dim, out_dim, node_emb_dim, time_emb_dim, continuous_time_emb_dim,
                                        node_num, graph_num, use_mask, mask_pool,
                                        refine_dynamic_node=refine_dynamic_node, refine_dynamic_graph=refine_dynamic_graph,
                                        corr_enhance_mode=corr_enhance_mode,
                                        device=device))
        self.device = device

    def forward(self,
                x_t, init_state, graph_list, node_emb=None, time_emb=None, continuous_time_emb=None,
                prev_dynamic_node_emb=None, prev_stce_p=None, prev_stce_g=None, prev_stce_m=None, z=None, time_step=None):
        current_input = x_t
        output_hidden = []
        for i, cell in enumerate(self.cells):
            state, z, prev_dynamic_node_emb, prev_stce_p, prev_stce_g, prev_stce_m = cell(current_input, init_state[i], graph_list, node_emb,
                                                                                          [time_emb[1], time_emb[0]],
                                                                                          continuous_time_emb,
                                                                                          prev_dynamic_node_emb,
                                                                                          prev_stce_p, prev_stce_g, prev_stce_m,
                                                                                          z=z, time_step=time_step)
            output_hidden.append(state)
            current_input = state

        return current_input, output_hidden, prev_dynamic_node_emb, prev_stce_p, prev_stce_g, prev_stce_m, z

class MetaDG(nn.Module):
    def __init__(self,
                 in_dim=3, out_dim=1, out_steps=12, node_num=None, adj_predef=None, hop_num_k=2,
                 node_emb_dim=16, tod_emb_dim=8, dow_emb_dim=8, continuous_time_emb_dim=8, hidden_dim=64, layer_num=1,
                 use_mask=True, refine_dynamic_node=True, refine_dynamic_graph=True, corr_enhance_mode='ST',
                 use_curriculum_learning=True, cl_decay_steps=4000, device=None):
        super(MetaDG, self).__init__()

        self.node_num = node_num
        self.adj_predef = adj_predef
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.out_steps = out_steps
        self.node_emb_dim = node_emb_dim
        self.layer_num = layer_num
        self.use_curriculum_learning = use_curriculum_learning
        self.cl_decay_steps = cl_decay_steps

        self.use_mask = use_mask
        self.refine_dynamic_node = refine_dynamic_node
        self.refine_dynamic_graph = refine_dynamic_graph

        self.device = device

        print(f'S-{node_emb_dim}-TOD-{tod_emb_dim}-DOW-{dow_emb_dim}-C-{continuous_time_emb_dim}')

        self.node_emb = nn.init.xavier_normal_(nn.Parameter(torch.empty(self.node_num, self.node_emb_dim)))
        self.tod_embedding = nn.Embedding(288, tod_emb_dim)
        self.dow_embedding = nn.Embedding(7, dow_emb_dim)

        use_dg = True  # Dynamic-Graph
        print('Use Dynamic Graph')
        use_dp = True                                                   # Dynamic-Parameters
        print('Use Dynamic MetaParam for each Time Step')

        use_mask = True
        print('Use Mask')
        # use_mask = False
        # print('No Mask')
        self.use_mask = use_mask
        if use_mask:
            time_emb_dim = continuous_time_emb_dim
            self.time_frequency_emb = nn.init.xavier_normal_(nn.Parameter(torch.empty(1, 1, time_emb_dim)))
            self.scale = (1 / time_emb_dim) ** 0.5
            mask_emb_dim = node_emb_dim * 4
            self.mask_pool = nn.init.xavier_normal_(nn.Parameter(torch.FloatTensor(time_emb_dim * 2, 1, node_emb_dim, mask_emb_dim)))
        else:
            self.mask_pool = None

        if refine_dynamic_node:
            print(f'with {corr_enhance_mode}CE')
        elif not refine_dynamic_node:
            print(f'w/o STCE')
        if refine_dynamic_graph:
            print(f'with DGQ')
        elif not refine_dynamic_graph:
            print(f'w/o DGQ')

        self.encoder = Encoder(hop_num_k=hop_num_k, in_dim=in_dim, hidden_dim=hidden_dim, out_dim=hidden_dim,
                               node_emb_dim=node_emb_dim, time_emb_dim=tod_emb_dim + dow_emb_dim, continuous_time_emb_dim=continuous_time_emb_dim,
                               node_num=node_num, graph_num=1, layer_num=self.layer_num, use_mask=use_mask, mask_pool=self.mask_pool,
                               refine_dynamic_node=refine_dynamic_node, refine_dynamic_graph=refine_dynamic_graph, corr_enhance_mode=corr_enhance_mode, device=self.device)

        self.decoder = Decoder(hop_num_k=hop_num_k, in_dim=in_dim, hidden_dim=hidden_dim, out_dim=hidden_dim,
                               node_emb_dim=node_emb_dim, time_emb_dim=tod_emb_dim + dow_emb_dim, continuous_time_emb_dim=continuous_time_emb_dim,
                               node_num=node_num, graph_num=1, layer_num=self.layer_num, use_mask=use_mask, mask_pool=self.mask_pool,
                               refine_dynamic_node=refine_dynamic_node, refine_dynamic_graph=refine_dynamic_graph, corr_enhance_mode=corr_enhance_mode, device=self.device)

        self.fc_final = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, y_cov, labels=None, batches_seen=None):
        b, in_step, n, _ = x.size()

        if self.use_mask:
            continuous_time = 2 * np.pi * (x[:, :, 0, 1] + x[:, :, 0, 2]) / 7  # range - (0, 1)
            continuous_time = self.time_frequency_emb * continuous_time.unsqueeze(-1)
            # continuous_time = (1 / self.time_frequency_emb) * continuous_time.unsqueeze(-1)
            time_cos = torch.cos(continuous_time)
            time_sin = torch.sin(continuous_time)
            continuous_time_emb = torch.stack([time_cos, time_sin], dim=-1).reshape(b, in_step, -1)
            continuous_time_emb = continuous_time_emb * self.scale
            # continuous_time_emb = self.compress_time_emb(continuous_time_emb)
        else:
            continuous_time_emb = None

        stepwise_tod = x[:, :, 0, 1]
        stepwise_dow = x[:, :, 0, 2]
        stepwise_tod_emb = self.tod_embedding((stepwise_tod * 288).long())  # (batch_size, time_step, tod_embedding_dim)
        stepwise_dow_emb = self.dow_embedding(stepwise_dow.long())  # (batch_size, time_step, dow_embedding_dim)
        stepwise_time_embedding = torch.cat([stepwise_tod_emb, stepwise_dow_emb], dim=-1)  # (B, T, tod+dow)

        node_emb = self.node_emb
        graph = torch.softmax(torch.relu(node_emb @ node_emb.T), dim=-1)  # torch.mm()

        h, _ = self.encoder(x, [graph], node_emb, stepwise_time_embedding, continuous_time_emb=continuous_time_emb, use_mask=self.use_mask)
        h_last = h[:, -1, :, :]

        state = [h_last] * self.layer_num
        go = torch.zeros((x.shape[0], self.node_num, self.out_dim), device=self.device)

        if self.use_mask:
            continuous_time = 2 * np.pi * (y_cov[:, :, 0, 0] + y_cov[:, :, 0, 1]) / 7  # range - (0, 1)
            continuous_time = self.time_frequency_emb * continuous_time.unsqueeze(-1)
            time_cos = torch.cos(continuous_time)
            time_sin = torch.sin(continuous_time)
            continuous_time_emb = torch.stack([time_cos, time_sin], dim=-1).reshape(b, in_step, -1)
            continuous_time_emb = continuous_time_emb * self.scale
        else:
            continuous_time_emb = None

        stepwise_tod = y_cov[:, :, 0, 0]
        stepwise_dow = y_cov[:, :, 0, 1]
        stepwise_tod_emb = self.tod_embedding((stepwise_tod * 288).long())  # (batch_size, time_step, tod_embedding_dim)
        stepwise_dow_emb = self.dow_embedding(stepwise_dow.long())  # (batch_size, time_step, dow_embedding_dim)
        stepwise_time_embedding = torch.cat([stepwise_tod_emb, stepwise_dow_emb], dim=-1)  # (B, T, tod+dow)

        out = []

        prev_dynamic_node_emb = prev_stce_p = prev_stce_g = prev_stce_m = node_emb
        z = None

        for t in range(self.out_steps):
            current_input = torch.cat([go, y_cov[:, t, ...]], dim=-1)
            time_emb = [stepwise_time_embedding[:, t, :], stepwise_time_embedding[:, -1, :]]
            if self.use_mask:
                continuous_time_emb_t = continuous_time_emb[:, t, ...]
            else:
                continuous_time_emb_t = None
            h_de, state, prev_dynamic_node_emb, prev_stce_p, prev_stce_g, prev_stce_m, z = self.decoder(current_input, state, [graph], node_emb=node_emb,
                                                                                                        time_emb=time_emb,
                                                                                                        continuous_time_emb=continuous_time_emb_t,
                                                                                                        prev_dynamic_node_emb=prev_dynamic_node_emb,
                                                                                                        prev_stce_p=prev_stce_p,
                                                                                                        prev_stce_g=prev_stce_g,
                                                                                                        prev_stce_m=prev_stce_m,
                                                                                                        z=z,
                                                                                                        time_step=t)
            go = self.fc_final(h_de)
            out.append(go)
            if self.training and self.use_curriculum_learning:
                c = np.random.uniform(0, 1)
                if c < self.compute_sampling_threshold(batches_seen):
                    go = labels[:, t, ...]

        output = torch.stack(out, dim=1)

        return output

    def compute_sampling_threshold(self, batches_seen):
        return self.cl_decay_steps / (self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))
