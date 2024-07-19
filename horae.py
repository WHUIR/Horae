import torch.nn.functional as F
from sequential_recommender import SequentialRecommender
from recbole.model.layers import TransformerEncoder
from torch_scatter import scatter_add
import torch
from torch import nn
from torch.nn import Transformer


class PWLayer(nn.Module):
    def __init__(self, input_size, output_size, dropout=0.0):
        super(PWLayer, self).__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.bias = nn.Parameter(torch.zeros(input_size), requires_grad=True)
        self.lin = nn.Linear(input_size, output_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, x):
        return self.lin(self.dropout(x) - self.bias)


class MoEAdaptorLayer(nn.Module):
    def __init__(self, n_exps, layers, dropout=0.0, noise=True):
        super(MoEAdaptorLayer, self).__init__()

        self.n_exps = n_exps
        self.noisy_gating = noise

        self.experts = nn.ModuleList([PWLayer(layers[0], layers[1], dropout) for i in range(n_exps)])
        self.w_gate = nn.Parameter(torch.zeros(layers[0], n_exps), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(layers[0], n_exps), requires_grad=True)

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        clean_logits = x @ self.w_gate
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = ((F.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits).to(x.device) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        gates = F.softmax(logits, dim=-1)
        return gates

    def forward(self, x):
        gates = self.noisy_top_k_gating(x, self.training)  # (B, n_E)
        expert_outputs = [self.experts[i](x).unsqueeze(-2) for i in range(self.n_exps)]  # [(B, 1, D)]
        expert_outputs = torch.cat(expert_outputs, dim=-2)
        multiple_outputs = gates.unsqueeze(-1) * expert_outputs
        return multiple_outputs.sum(dim=-2)


class TimeAwareAttention(nn.Module):
    def __init__(self, config):
        super(TimeAwareAttention, self).__init__()
        self.hidden_size = config['hidden_size']  # same as embedding_size
        self.max_seq_length = config['max_seq_len']
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.time_diff_embedding_t = nn.Embedding(14, self.hidden_size) #13 is padding
        self.time_diff_embedding_n = nn.Embedding(14, self.hidden_size) #13 is padding
        self.attn_linear = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Dropout(p=config['hidden_dropout_prob']),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, 1)
        )
        self.tau = 1

    def forward(self, seq_output, item_seq, time_seq):

        position_ids = torch.arange(seq_output.size(1), dtype=torch.long, device=seq_output.device)
        position_ids = position_ids.view(1, seq_output.size(1), 1).expand(seq_output.size(0), -1, -1) #(batch, seqlen, 1)

        time_diff_target = time_seq[:, :, 0]
        time_diff_neighbor = time_seq[:, :, 1]
        position_embedding = self.position_embedding(position_ids)

        time_diff_embedding = self.time_diff_embedding_t(time_diff_target).unsqueeze(-2) + self.time_diff_embedding_n(time_diff_neighbor).unsqueeze(-2)
        tma_inputs = position_embedding + seq_output + time_diff_embedding

        tma_weight = self.attn_linear(tma_inputs).squeeze(-1) / self.tau #(batch, seqlen, 1)

        tma_weight = torch.masked_fill(tma_weight, (item_seq == 0), -1e9)
        tma_weight = F.softmax(tma_weight.view(seq_output.size(0), -1), dim=-1)
        return tma_weight.view(seq_output.size(0), seq_output.size(1), seq_output.size(2))

class MultiInterestExtractor(nn.Module):
    def __init__(self, config):
        super(MultiInterestExtractor, self).__init__()

        self.hidden_size = config['hidden_size']
        self.initializer_range = config['initializer_range']
        self.linear = nn.Linear(self.hidden_size, self.hidden_size)
        self.aspect_embs = nn.Embedding(config['aspects'], config['hidden_size'])

        self.time_aware_attn = TimeAwareAttention(config)

        self.aspects = config['aspects']
        self.tau = 1
        self.noise_scale = config['noise_scale']
        self.device = config['device']
        self.caps_layers = config['caps_layers']
        self.moe_dropout = nn.Dropout(p=config['moe_dropout'])
        self.ln = nn.LayerNorm(self.hidden_size, eps=1e-12)

    def forward_sequence(self, item_emb, item_seq, time_seq, aspect_mask=None, topk_gates_idx=None):
        batch_size, seq_len = item_emb.size()[0], item_emb.size()[1]
        tma_weight = self.time_aware_attn(item_emb.unsqueeze(-2), item_seq.unsqueeze(-1), time_seq)
        # tma_weight (batch, seqlen, 1)

        gates = self.generate_gates(item_emb)

        if aspect_mask is None:
            topk_gates, topk_gates_idx = torch.topk(gates, dim=-1, k=1)  # (batch, seqlen, 1)
            src = torch.ones([batch_size, seq_len], device=self.device)
            src = torch.masked_fill(src, item_seq.unsqueeze(-1).view(batch_size, -1) == 0,
                                    0) #mask padding
            aspect_mask = scatter_add(src,
                                      topk_gates_idx.view(batch_size, -1),
                                      out=torch.zeros([batch_size, self.aspects], device=self.device))
            aspect_mask = aspect_mask == 0

        else:
            aspect_mask = aspect_mask
            topk_gates_idx = topk_gates_idx

        item_moe_emb = F.tanh(self.linear(item_emb)) + item_emb
        item_moe_emb = self.ln(self.moe_dropout(item_moe_emb))
        item_moe_emb = item_moe_emb.unsqueeze(2) #(batch, seqlen, 1, emb)

        bij = gates  #(batch, seqlen, aspects)
        interest_capsule = self.aspect_embs.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        for i in range(self.caps_layers):
            seq_mask = item_seq == 0
            cij = torch.masked_fill(bij, aspect_mask.unsqueeze(1), -1e9)
            cij = torch.softmax(cij / self.tau, dim=-1)
            cij = torch.masked_fill(cij, seq_mask.unsqueeze(-1), 0)
            interest_capsule = torch.sum(
                cij.unsqueeze(-1) * item_moe_emb * tma_weight.unsqueeze(-1),
                dim=1)
            cap_norm = torch.sum(torch.pow(interest_capsule, 2), dim=-1, keepdim=True)
            scalar_factor = cap_norm / (1 + cap_norm) / torch.sqrt(cap_norm + 1e-9)
            interest_capsule = scalar_factor * interest_capsule

            # Squash
            delta_weight = (item_moe_emb * interest_capsule.unsqueeze(1)).sum(dim=-1)
            bij = bij + delta_weight




        expanded_topk_gates_idx = topk_gates_idx.expand(-1, -1, interest_capsule.size(-1))
        activated_interests = torch.gather(interest_capsule, 1, expanded_topk_gates_idx) #(batch, seqlen, emb)

        seq_positions = torch.arange(seq_len, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1).to(
            item_seq.device) + 1
        seq_positions = seq_positions.long()
        seq_positions = torch.where(item_seq == 0, torch.zeros_like(item_seq), seq_positions)
        seq_positions = seq_positions.unsqueeze(-1)  # Shape: (batch, seqlen, 1)
        activation_matrix = torch.full((batch_size, seq_len, interest_capsule.size()[1]), -1, dtype=torch.float32).to(item_seq.device)
        activation_matrix = activation_matrix.scatter(2, topk_gates_idx, seq_positions.float())
        seq_mask = item_seq == 0
        activation_matrix = torch.where(seq_mask.unsqueeze(-1), torch.full_like(activation_matrix, -1),
                                        activation_matrix)
        max_activation_pos, _ = activation_matrix.max(dim=1)  # Shape: (batch, aspects)
        aspects_item_idx = torch.where(max_activation_pos > -1, max_activation_pos - 1, max_activation_pos)

        return interest_capsule, F.softmax(gates / self.tau, dim=-1), aspect_mask, activated_interests, aspects_item_idx, topk_gates_idx

    def generate_gates(self, item_emb, noise_epsilon=1e-2):
        clean_gates = item_emb @ self.aspect_embs.weight.t()
        if self.training:
            noise_stddev = clean_gates.detach() * self.noise_scale + noise_epsilon
            noisy_gates = clean_gates + (torch.randn_like(clean_gates).to(item_emb.device) * noise_stddev)
            gates = noisy_gates
        else:
            gates = clean_gates
        return gates

    def forward_item(self, item_emb):
        if len(item_emb.size()) == 2:
            item_emb = item_emb.unsqueeze(1)
        gates = self.generate_gates(item_emb)
        return item_emb.unsqueeze(-2), F.softmax(gates / self.tau, dim=-1)

    def forward_item_all(self):
        pass
class WeightedGRU(nn.Module):
    def __init__(self, config):
        super(WeightedGRU, self).__init__()
        input_dim = config['hidden_size']
        hidden_dim = config['hidden_size']
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, hidden_dim) #wise

    def forward(self, item_seq, interest_seq):
        gru_output, _ = self.gru(item_seq)
        weights = self.fc(gru_output) #(batch, seqlen, emb)
        weights = torch.sigmoid(weights)
        scaled_interest_seq = weights * interest_seq
        return scaled_interest_seq

class Horae(SequentialRecommender):
    def __init__(self, config):
        super(Horae, self).__init__(config)

        # load parameters info
        self.hidden_size = config['hidden_size']
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.layer_norm_eps = config['layer_norm_eps']
        self.n_items = config['item_count']
        self.max_seq_length = config['max_seq_len']
        self.stage = config['stage']

        self.initializer_range = config['initializer_range']
        self.loss_type = config['loss_type']

        # define layers and loss
        if self.stage == 'trans':
            self.item_embedding = nn.Embedding(self.n_items, self.hidden_size)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.plm_embedding = config['text_emb'].to(self.device)
        self.time_diff_embedding_t = nn.Embedding(14, self.hidden_size)
        self.time_diff_embedding_n = nn.Embedding(14, self.hidden_size)

        self.trm_encoder = TransformerEncoder(
            n_layers=config['encoder_layers'],
            n_heads=config['n_heads'],
            hidden_size=self.hidden_size,
            inner_size=config['inner_size'],
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=config['attn_dropout_prob'],
            hidden_act=config['hidden_act'],
            layer_norm_eps=self.layer_norm_eps
        )

        self.trm = TransformerEncoder(
            n_layers=config['n_layers'],
            n_heads=config['n_heads'],
            hidden_size=self.hidden_size,
            inner_size=config['inner_size'],
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=config['attn_dropout_prob'],
            hidden_act=config['hidden_act'],
            layer_norm_eps=self.layer_norm_eps
        )
        self.gated = WeightedGRU(config)

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        self.act = nn.LeakyReLU()

        self.mi_extractor = MultiInterestExtractor(config)
        self.pad_mode = config['pad_mode']
        self.neg_count = config['neg_count']
        self.item_embs = None
        self.balance_alpha = config['balance_alpha']
        self.aspects = config['aspects']
        self.time_aware_attn = TimeAwareAttention(config)
        self.moe_dropout = nn.Dropout(p=config['moe_dropout'])
        self.aspect_cons_tau = config['aspect_cons_tau']
        self.aspect_alpha = config['aspect_alpha']

        self.moe_adaptor = MoEAdaptorLayer(
            config['n_exps'],
            config['adaptor_layers'],
            config['adaptor_dropout_prob']
        )
        self.mask_idx = -1
        self.mask_param = nn.Parameter(torch.zeros(config['hidden_size']).normal_(0, self.initializer_range),
                                       requires_grad=True)
        self.mask_param2 = nn.Parameter(torch.zeros(config['hidden_size']).normal_(0, self.initializer_range),
                                       requires_grad=True)

        self.seq_cons_alpha = config['seq_cons_alpha']
        self.seq_cons_tau = config['seq_cons_tau']

        self.cal_triplet_alpha = config['triplet_alpha']
        self.cal_triplet_beta = config['triplet_beta']
        self.center_alpha = config['center_alpha']

        # parameters initialization
        self.apply(self._init_weights)
        self.config = config
        self.item_drop_ratio = 0.2
        self.interest_drop_ratio = 0.2

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def seq_encode_t(self, item_seq, item_emb, item_seq_len, time_seq):
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.view(1, item_seq.size(1)).expand(item_seq.size(0), -1) #(batch, seqlen, 1)

        time_diff_target = time_seq[:, :, 0]
        time_diff_neighbor = time_seq[:, :, 1]
        position_embedding = self.position_embedding(position_ids) #(batch, seqlen, 1, emb)

        time_diff_embedding = self.time_diff_embedding_t(time_diff_target) + self.time_diff_embedding_n(time_diff_neighbor)
        input_emb = position_embedding + item_emb + time_diff_embedding  # (batch, seqlen, 1, emb)

        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq, bidirectional=True)
        trm_output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
        output_emb = trm_output[-1]
        return output_emb


    def ffn_trm(self, interests_activate, item_emb, item_seq, time_seq):
        interests_residual = interests_activate.to(self.device) #(batch, seqlen, emb)
        input_emb = interests_residual
        extended_attention_mask = self.get_attention_mask(item_seq, bidirectional=False)
        output = self.trm(input_emb, extended_attention_mask, output_all_encoded_layers=True) #encoder
        return output[-1]


    def encoder(self, interests_activate, item_seq, item_emb, time_seq=None):
        interests_seq = self.ffn_trm(interests_activate, item_emb, item_seq, time_seq)
        interests_seq = self.gated(item_emb, interests_seq)

        return interests_seq

    def get_updated_interests(self, interests_seq, aspects_item_idx):
        batch_size, _, emb = interests_seq.size()
        interests_seq, aspects_item_idx = interests_seq.to(self.device), aspects_item_idx.to(self.device)
        padding = torch.zeros((batch_size, 1, emb)).to(self.device)
        interest_seq = torch.cat([padding, interests_seq], dim=1)
        gather_indices = aspects_item_idx + 1
        gather_indices = gather_indices.unsqueeze(-1).repeat(1, 1, emb).long().to(self.device)
        aspects_interests = torch.gather(interest_seq, 1, gather_indices).to(self.device)
        aspects_interests = torch.where(aspects_item_idx.unsqueeze(-1) == -1, torch.zeros_like(aspects_interests),
                                        aspects_interests).to(self.device)

        return aspects_interests

    def calculate_time_diff_target(self, time_seq, target_time, item_seq):
        target_time_expanded = target_time.unsqueeze(1).expand_as(time_seq)
        time_diff_seconds = target_time_expanded - time_seq
        scale = self.config['time_scale']
        time_diff_days = time_diff_seconds / scale

        bins = torch.tensor([1, 2, 4, 7, 14, 30, 60, 120, 180, 365, 730, 1825], dtype=torch.float).to(self.device)
        bins = bins.view(1, 1, -1)  #(1, 1, 12)

        condition = time_diff_days.unsqueeze(-1) < bins
        time_diff_days_binned = condition.sum(dim=-1)
        time_diff_days_binned = len(bins[0, 0]) - time_diff_days_binned

        time_diff_days_binned[time_diff_days >= bins[0, 0, -1]] = len(bins[0, 0]) + 1  # >5year 12
        mask = item_seq == 0
        time_diff_days_binned = torch.masked_fill(time_diff_days_binned, mask, 13)

        return time_diff_days_binned


    def calculate_time_diff_neighbor(self, time_seq, target_time, item_seq):
        next_time_seq = torch.roll(time_seq, shifts=-1, dims=1)
        next_time_seq[:, -1] = target_time
        time_diff_seconds = next_time_seq - time_seq
        scale = self.config['time_scale']
        time_diff_days = time_diff_seconds / scale

        bins = torch.tensor([1, 2, 4, 7, 14, 30, 60, 120, 180, 365, 730, 1825], dtype=torch.float).to(self.device)
        bins = bins.view(1, 1, -1)

        condition = time_diff_days.unsqueeze(-1) < bins
        time_diff_days_binned = condition.sum(dim=-1)
        time_diff_days_binned = len(bins[0, 0]) - time_diff_days_binned

        time_diff_days_binned[time_diff_days >= bins[0, 0, -1]] = len(bins[0, 0]) + 1  # >5year 12
        mask = item_seq == 0
        time_diff_days_binned = torch.masked_fill(time_diff_days_binned, mask, 13)
        return time_diff_days_binned

    def calculate_time_diff(self, time_seq, target_time, item_seq):
        time_diff_target = self.calculate_time_diff_target(time_seq, target_time, item_seq)
        time_diff_neighbor = self.calculate_time_diff_neighbor(time_seq, target_time, item_seq)
        time_diff = torch.cat([time_diff_target.unsqueeze(-1), time_diff_neighbor.unsqueeze(-1)], dim=-1) #(batch, seqlen, 2)
        return time_diff

    def forward(self, item_seq, item_seq_len, time_seq):
        item_emb = self.moe_adaptor(self.plm_embedding[item_seq].to(self.device))
        if self.stage == 'trans':
            item_emb = item_emb + self.item_embedding(item_seq)
        # Sequential
        seq_output = self.seq_encode_t(item_seq, item_emb, item_seq_len, time_seq) #(batch, seqlen, emb)
        # MoE
        interests, gates, aspect_mask, interests_activate, aspects_item_idx, topk_gates_idx = self.mi_extractor.forward_sequence(seq_output,
                                                                           item_seq, time_seq)
        interests_seq = self.encoder(interests_activate, item_seq, seq_output, time_seq) #(batch, seqlen, emb)
        aspects_interests = self.get_updated_interests(interests_seq, aspects_item_idx.clone())
        return interests, gates, aspect_mask, aspects_interests, topk_gates_idx

    def calculate_loss(self, interaction):
        if self.stage == 'pretrain':
            return self.calculate_pretrain_loss(interaction)
        else:
            return self.calculate_downstream_loss(interaction)

    def calculate_pretrain_loss(self, interaction):
        batch_size = interaction['item_seqs'].size()[0]
        item_seq = interaction['item_seqs']
        item_seq_len = interaction['lengths']
        time_seq = interaction['timestamps']
        target_time = interaction['target_timestamp']
        time_seq = self.calculate_time_diff(time_seq, target_time, item_seq)

        interests, moe_gates, aspect_mask, interests_new, topk_gates_idx = self.forward(item_seq, item_seq_len, time_seq)
        pos_items = interaction['labels']
        total_embs, item_gates = self.mi_extractor.forward_item(
            self.moe_adaptor(self.plm_embedding[pos_items].to(self.device))) #(batch, 1, 1, emb), (batch, 1, aspects)
        total_embs = total_embs.squeeze(1)


        total_score = interests_new.transpose(0, 1) @ total_embs.permute(1, 2, 0) #(aspects, batch, batch) or (seqlen, batch, batch)
        total_score = torch.masked_fill(total_score, aspect_mask.t().unsqueeze(-1), -100)
        max_score, max_idx = total_score.max(dim=0) #(batch, batch)
        loss = F.cross_entropy(max_score,
                               torch.arange(total_embs.size()[0], device=self.device)) #lrec

        balance_loss = self.cal_balance_loss(moe_gates, item_seq) + self.cal_balance_loss(item_gates,
                                                                                          torch.ones(size=[batch_size],
                                                                                                     device=self.device)) #coverage regularization, lb
        aspect_contrastive_loss = self.cal_aspects_contrastive_loss() #capsule regularization, lc

        mi_cs_loss = self.cal_mI_contrastive_loss(interaction, interests_new, aspect_mask, moe_gates, topk_gates_idx) #interest-level contrastive learning, li

        total_loss = loss + self.balance_alpha * balance_loss + self.aspect_alpha * aspect_contrastive_loss + self.seq_cons_alpha * mi_cs_loss

        triplet_loss = self.cal_triplet_loss(interaction, interests, interests_new, max_idx, aspect_mask, topk_gates_idx, moe_gates, total_embs)
        total_loss += self.cal_triplet_alpha * triplet_loss

        center_loss = self.cal_center_loss(interaction, interests, aspect_mask, moe_gates)
        total_loss += self.center_alpha * center_loss

        return [total_loss]

    def cal_center_loss(self, interaction, interests, aspect_mask, gates):
        batch_size, _, emb = interests.size()
        aspect = aspect_mask.size()[1]
        item_seq = interaction['item_seqs']
        time_seq = interaction['timestamps']
        target_time = interaction['target_timestamp']
        time_seq = self.calculate_time_diff(time_seq, target_time, item_seq)
        seqlen = item_seq.size(-1)
        item_emb = self.moe_adaptor(self.plm_embedding[item_seq].to(self.device))
        seq_output = self.seq_encode_t(item_seq, item_emb, seqlen, time_seq)
        _, topk_gates_idx = torch.topk(gates, dim=-1, k=1)  # (batch, seqlen, 1)

        gates_expanded = topk_gates_idx.expand(-1, -1, interests.size(-1)) 
        interests_expanded = interests.unsqueeze(1).expand(batch_size, seqlen, aspect, emb) 

        gathered_interest = torch.gather(interests_expanded, 2,
                                         gates_expanded.unsqueeze(2).expand(batch_size, seqlen, 1, emb)).squeeze(2) #(batch, seqlen, emb)


        distance = torch.norm(seq_output - gathered_interest.squeeze(2), p=2, dim=-1,
                              keepdim=True)  # (batch, seqlen, 1)


        topk_gates_idx_expanded = topk_gates_idx.expand(batch_size, seqlen, aspect)  # (batch, seqlen, aspect)
        matching_matrix = topk_gates_idx_expanded == torch.arange(aspect, device=self.device).view(1, 1, aspect)  # (batch, seqlen, aspect)
        aspect_count = torch.sum(matching_matrix, dim=1, dtype=torch.float)  # (batch, aspect)
        mean_distance = torch.sum(distance.squeeze(2)[:, :, None] * matching_matrix, dim=1) / (
                    aspect_count + 1e-8)

        mean_distance_activated = mean_distance.masked_select(~aspect_mask)

        if mean_distance_activated.numel() > 0:
            center_loss = mean_distance_activated.mean()
        else:
            center_loss = torch.tensor(0.0, device=self.device)

        return center_loss

    def calculate_downstream_loss(self, interaction):
        item_seq = interaction['item_seqs']
        item_seq_len = interaction['lengths']
        time_seq = interaction['timestamps']
        target_time = interaction['target_timestamp']
        time_seq = self.calculate_time_diff(time_seq, target_time, item_seq)
        interests, moe_gates, aspect_mask, interests_new,topk_gates_idx = self.forward(item_seq, item_seq_len, time_seq)
        total_embs = self.moe_adaptor(self.plm_embedding.to(self.device))
        if self.stage == 'trans':
            total_embs = total_embs + self.item_embedding.weight
        total_embs, item_gates = self.mi_extractor.forward_item(total_embs)
        total_embs = total_embs.squeeze(1)


        total_score = interests_new.transpose(0, 1) @ total_embs.permute(1, 2, 0)
        total_score = total_score.permute(1, 2, 0)
        total_score = torch.masked_fill(total_score, aspect_mask.unsqueeze(1), -100)
        total_score, max_idx = torch.max(total_score, dim=-1)

        loss = F.cross_entropy(total_score,
                               interaction['labels'])
        balance_loss = self.cal_balance_loss(moe_gates, item_seq) + self.cal_balance_loss(item_gates, torch.ones(
            size=[item_gates.size()[0]], device=self.device))
        aspect_contrastive_loss = self.cal_aspects_contrastive_loss()

        total_loss = loss + self.balance_alpha * balance_loss + self.aspect_alpha * aspect_contrastive_loss

        mi_cs_loss = 0
        is_cs_loss = 0
        triplet_loss = 0
        return [total_loss]

    def predict(self, interaction):
        item_seq = interaction['item_seqs']
        item_seq_len = interaction['lengths']
        time_seq = interaction['timestamps']
        target_time = interaction['target_timestamp']
        time_seq = self.calculate_time_diff(time_seq, target_time, item_seq)
        interests, moe_gates, aspect_mask, interests_new,_ = self.forward(item_seq, item_seq_len, time_seq)

        pos_items = interaction['labels'] #(batch,)
        neg_items = interaction['neg_items'] #(batch, 100)

        total_items = torch.cat([pos_items.unsqueeze(-1), neg_items], dim=1)
        total_embs, item_gates = self.mi_extractor.forward_item(
            self.moe_adaptor(self.plm_embedding[total_items].to(self.device)))


        total_score = (interests_new.unsqueeze(1) * total_embs).sum(dim=-1)
        mi_logits = torch.masked_fill(total_score, aspect_mask.unsqueeze(1), -100)
        logits = torch.max(mi_logits, dim=-1)[0]

        return logits, torch.zeros(interaction['labels'].size()[0])

    def full_sort_predict(self, interaction):
        item_seq = interaction['item_seqs']
        item_seq_len = interaction['lengths']
        time_seq = interaction['timestamps']
        target_time = interaction['target_timestamp']
        time_seq = self.calculate_time_diff(time_seq, target_time, item_seq)
        interests, moe_gates, aspect_mask, interests_new,_ = self.forward(item_seq, item_seq_len, time_seq)
        #(batch, aspects, emb)

        total_embs = self.moe_adaptor(self.plm_embedding.to(self.device)) #(item_num, emb)
        if self.stage == 'trans':
            total_embs = total_embs + self.item_embedding.weight
        total_embs, item_gates = self.mi_extractor.forward_item(total_embs)
        total_embs = total_embs.squeeze(1)

        total_score = interests_new.transpose(0, 1) @ total_embs.permute(1, 2, 0)
        total_score = total_score.permute(1, 2, 0) #(batch, batch, aspects)
        total_score = torch.masked_fill(total_score, aspect_mask.unsqueeze(1), -100)
        total_score = torch.max(total_score, dim=-1)[0] #(batch, batch)

        return total_score, interaction['labels']


    def cal_balance_loss(self, gates, item_seq): #coverage regularization
        gates = gates.view(-1, gates.size()[-1])
        gates = gates[item_seq.view(-1) != 0]

        _, idx = gates.max(dim=-1)

        p = gates.mean(dim=0)
        f = scatter_add(torch.ones(size=idx.size(), device=self.device), idx,
                        out=torch.zeros(size=p.size(), device=self.device)) / gates.size()[0]
        return (f * p).sum()

    def cal_aspects_contrastive_loss(self):
        embs = self.mi_extractor.aspect_embs.weight
        embs = F.normalize(embs, dim=-1)
        sim = embs @ embs.t()
        sim = sim / self.aspect_cons_tau
        loss = F.cross_entropy(sim, torch.arange(embs.size()[0], device=self.device))
        return loss

    def cal_mI_contrastive_loss(self, interaction, interests, aspect_mask, gates, topk_gates_idx): #Interest-Level Pre-training
        item_seq, item_seq_len = interaction['item_seqs'], interaction['lengths']
        time_seq = interaction['timestamps']
        target_time = interaction['target_timestamp']
        time_seq = self.calculate_time_diff(time_seq, target_time, item_seq)
        item_seq_aug, item_seq_len_aug, seq_aug_mask = self.seq_aug(item_seq, item_seq_len)
        item_emb_aug = self.moe_adaptor(
            self.plm_embedding[item_seq_aug].to(self.device))
        item_emb_aug[item_seq_aug == self.mask_idx] = self.mask_param.data
        seq_output_aug = self.seq_encode_t(item_seq_aug, item_emb_aug,
                                         item_seq_len_aug, time_seq)
        interests_aug, gates_aug, aspect_mask_aug, interests_aug_activate, aspects_item_idx,_ = self.mi_extractor.forward_sequence(seq_output_aug,
                                                                                       item_seq, time_seq,
                                                                                       aspect_mask,topk_gates_idx)
        interests_seq = self.encoder(interests_aug_activate, item_seq_aug, seq_output_aug, time_seq) #(batch, seqlen, emb)
        aspects_interests_aug = self.get_updated_interests(interests_seq, aspects_item_idx.clone()) #(batch, aspects, emb)
        #item_seq_aug: (batch, seqlen)
        mask_item_gates = gates[item_seq_aug == self.mask_idx] 
        _, mask_item_gates_max_idx = mask_item_gates.max(dim=-1) #(total_mask,)
        row_idx = (item_seq_aug == self.mask_idx).nonzero()[:, 0] #(total_mask,)

        interests = interests[row_idx, mask_item_gates_max_idx]
        interests_aug = aspects_interests_aug[row_idx, mask_item_gates_max_idx]

        interests_sim = interests @ interests_aug.t()
        mi_cs_loss = F.cross_entropy(interests_sim, torch.arange(interests_sim.size()[0], device=self.device))
        return mi_cs_loss

    def cal_triplet_loss(self, interaction, interests_old, interests_new, max_idx, aspect_mask, topk_gates_idx, gates, total_embs):
        item_seq, item_seq_len = interaction['item_seqs'], interaction['lengths']
        time_seq = interaction['timestamps']
        target_time = interaction['target_timestamp']
        time_seq = self.calculate_time_diff(time_seq, target_time, item_seq)
        batch_size, seq_len, _ = topk_gates_idx.size()
        topk_gates_idx, gates = topk_gates_idx.to(self.device), gates.to(self.device)
        item_seq, interests_old, interests_new = item_seq.to(self.device), interests_old.to(self.device), interests_new.to(self.device)
        expanded_topk_gates_idx = topk_gates_idx.expand(-1, -1, interests_old.size(-1))
        activated_interests = torch.gather(interests_old, 1, expanded_topk_gates_idx) #(batch, seqlen, emb)

        seq_positions = torch.arange(seq_len, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1).to(
            self.device) + 1
        seq_positions = seq_positions.long()
        seq_positions = torch.where(item_seq == 0, torch.zeros_like(item_seq), seq_positions)
        seq_positions = seq_positions.unsqueeze(-1)
        activation_matrix = torch.full((batch_size, seq_len, gates.size()[2]), -1, dtype=torch.float32).to(self.device)
        activation_matrix = activation_matrix.scatter(2, topk_gates_idx, seq_positions.float())
        seq_mask = item_seq == 0
        activation_matrix = torch.where(seq_mask.unsqueeze(-1), torch.full_like(activation_matrix, -1),
                                        activation_matrix)

        max_activation_pos, _ = activation_matrix.max(dim=1)

        aspects_item_idx = torch.where(max_activation_pos > -1, max_activation_pos - 1, max_activation_pos)


        #MASK2
        item_seq_aug, item_seq_len_aug, seq_aug_mask = self.seq_aug(item_seq, item_seq_len)
        item_emb_aug = self.moe_adaptor(
            self.plm_embedding[item_seq_aug].to(self.device))
        item_emb_aug[item_seq_aug == self.mask_idx] = self.mask_param.data
        seq_output_aug = self.seq_encode_t(item_seq_aug, item_emb_aug,
                                         item_seq_len_aug, time_seq)

        activated_interests[item_seq_aug == self.mask_idx] = self.mask_param2.data

        interests_seq_aug = self.encoder(activated_interests, item_seq_aug, seq_output_aug, time_seq)
        interests_aug = self.get_updated_interests(interests_seq_aug, aspects_item_idx.clone())

        if self.config['stage'] == 'pretrain':

            diagonal_indices = torch.diag(max_idx).to(self.device)
            batch_indices = torch.arange(len(diagonal_indices)).to(self.device)
            interests_old_ = interests_old[batch_indices, diagonal_indices].squeeze(1)
            interests_new_ = interests_new[batch_indices, diagonal_indices].squeeze(1)
            interests_aug_ = interests_aug[batch_indices, diagonal_indices].squeeze(1)
        else:
            pos_items = interaction['labels']
            batch_indices = torch.arange(pos_items.size()[0]).to(self.device)
            gate_idx_for_labels = max_idx[batch_indices, interaction['labels']]
            # max_idx (batch, item_num)
            interests_old_ = interests_old[batch_indices, gate_idx_for_labels].squeeze(1)
            interests_new_ = interests_new[batch_indices, gate_idx_for_labels].squeeze(1)
            interests_aug_ = interests_aug[batch_indices, gate_idx_for_labels].squeeze(1)


        total_embs = total_embs.squeeze()  # shape: (batch, emb)

        # Compute dot products for pos, anchor, neg with all items
        pos_scores = interests_new_.matmul(total_embs.t())
        anchor_scores = interests_aug_.matmul(total_embs.t())
        neg_scores = interests_old_.matmul(total_embs.t())

        # Extract the probabilities of the label item
        pos_probs = F.softmax(pos_scores, dim=1)
        anchor_probs = F.softmax(anchor_scores, dim=1)
        neg_probs = F.softmax(neg_scores, dim=1)

        # Extract the probabilities of the label item (the diagonal contains the correct probs)
        pos_label_prob = torch.diag(pos_probs)
        anchor_label_prob = torch.diag(anchor_probs)
        neg_label_prob = torch.diag(neg_probs)

        # Compute the loss
        loss_pos_anchor = -torch.log(F.sigmoid(pos_label_prob - anchor_label_prob)).mean()
        loss_anchor_neg = -torch.log(F.sigmoid(anchor_label_prob - neg_label_prob)).mean()

        total_score = interests_old.transpose(0, 1) @ total_embs.unsqueeze(1).permute(1, 2, 0)
        total_score = total_score.permute(1, 2, 0)
        total_score = torch.masked_fill(total_score, aspect_mask.unsqueeze(1), -100)
        total_score, _ = torch.max(total_score, dim=-1)
        loss_old = F.cross_entropy(total_score, torch.arange(total_embs.size()[0], device=self.device))

        triplet_loss = loss_pos_anchor + loss_anchor_neg + self.cal_triplet_beta * loss_old

        return triplet_loss


    def downstream_freeze_parameter(self):
        for _ in self.position_embedding.parameters():
            _.requires_grad = False

        for _ in self.trm_encoder.parameters():
            _.requires_grad = False

        for _ in self.trm.parameters():
            _.requires_grad = False

        for _ in self.time_diff_embedding_t.parameters():
            _.requires_grad = False
        for _ in self.time_diff_embedding_n.parameters():
            _.requires_grad = False

        for _ in self.gated.parameters():
            _.requires_grad = False


    def batch_step(self):
        self.n_batch += 1
        self.mi_extractor.tau = 1

    def seq_aug(self, item_seq, item_seq_len):
        item_seq = item_seq.cpu()
        item_seq_len = item_seq_len.cpu()
        mask_p = torch.full_like(item_seq, self.item_drop_ratio, dtype=torch.float)
        mask = torch.bernoulli(mask_p).to(torch.bool)
        mask[:, -1] = False
        mask = torch.masked_fill(mask, item_seq == 0, False)
        mask[item_seq_len < 5] = False
        item_seq_aug = torch.masked_fill(item_seq, mask, self.mask_idx)  # -1 represents [mask]
        item_seq_len_aug = (item_seq_aug != 0).sum(dim=-1)
        return item_seq_aug.to(self.device), item_seq_len_aug.to(self.device), mask
