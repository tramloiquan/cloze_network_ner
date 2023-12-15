import torch
import numpy as np
from typing import List, Optional
from crf import CRF
###
def positional_embedding(emb_dim, start_pos, end_pos):
    pe = []
    for pos in range(start_pos, end_pos+1):
        a=np.arange(emb_dim).astype(float)
        a1 = a[a%2==0].astype(int)
        a2 = a[a%2==1].astype(int)
        a[a1] = np.sin(pos/10000**(a1/emb_dim))
        a[a2] = np.cos(pos/10000**(2*(a2//2)/emb_dim))
        pe.append(a)
    #
    pe = torch.Tensor(np.stack(pe, axis=0))
    return pe
#
class MultiHeadAtt(torch.nn.Module):
    # pylint: disable=too-many-instance-attributes
    # pylint: disable=too-few-public-methods
    """
    There are two sub-blocks: attention and ffw
    There is layernorm before each sub-block, and residual layer for each sub-block
    """
    def __init__(self, q_input_dim: int, k_input_dim:int, v_input_dim:int,
            att_output_dim: int, num_heads: int):
        #
        super().__init__()
        #
        assert att_output_dim % num_heads == 0
        #
        self.num_heads = num_heads
        self.att_head_dim = int(att_output_dim/self.num_heads)
        ### attention
        self.query_layer = torch.nn.Linear(q_input_dim, self.att_head_dim * self.num_heads)
        self.key_layer = torch.nn.Linear(k_input_dim, self.att_head_dim * self.num_heads)
        self.value_layer = torch.nn.Linear(v_input_dim, self.att_head_dim * self.num_heads)
    #
    def forward(self, Q, K, V, mask=False):
        """
        Q,K,V: shape (batch_size, seq_size, att_input_dim)
        Return:
            ts05C_ctx: shape (batch_size, seq_size, att_output_dim)
        """
        batch_size=Q.shape[0]
        ### get query matrix
        ### shape (batch_size, num_heads, seq_size, att_head_dim)
        ts03A_query = (self.query_layer(Q)
            .reshape(batch_size, -1, self.num_heads, self.att_head_dim)
            .permute(0,2,1,3))
        ### get key matrix
        ### shape (batch_size, num_heads, seq_size, att_head_dim)
        ts03D_key = (self.key_layer(K)
            .reshape(batch_size, -1, self.num_heads, self.att_head_dim)
            .permute(0,2,1,3))
        ### get value matrix
        ts03F_value = (self.value_layer(V)
            .reshape(batch_size, -1, self.num_heads, self.att_head_dim)
            .permute(0,2,1,3))
        ## use layer_norm in similarity matrix to prevent exploding
        ts05A = (torch.matmul(ts03A_query, ts03D_key.transpose(2,3))
            / torch.sqrt(torch.tensor(self.att_head_dim))
            )
        if mask:
            ts02A_neg_inf = torch.full(ts05A.shape, -np.inf)
            ts02D_mask = torch.triu(ts02A_neg_inf, diagonal=1)
            ts05A = ts05A + ts02D_mask
        #
        ts05B_score = torch.nn.functional.softmax(ts05A, dim=-1)
        ts05C_ctx = (torch.matmul(ts05B_score, ts03F_value) ## shape (batch_size, num_head, seq_size, self.att_head_dim)
            .permute(0,2,1,3) ## shape (batch_size, seq_size, num_head, self.att_head_dim)
            .reshape(batch_size, -1, self.num_heads * self.att_head_dim)
            )
        ###
        return ts05C_ctx
#
class MultiHeadSelfAtt(torch.nn.Module):
    # pylint: disable=too-many-instance-attributes
    # pylint: disable=too-few-public-methods
    """
    There are two sub-blocks: attention and ffw
    There is layernorm before each sub-block, and residual layer for each sub-block
    """
    def __init__(self, att_input_dim: int, att_output_dim: int, num_heads: int):
        #
        super().__init__()
        #
        assert att_output_dim % num_heads == 0
        #
        self.num_heads = num_heads
        self.att_head_dim = int(att_output_dim/self.num_heads)
        ### attention
        self.query_layer = torch.nn.Linear(att_input_dim, self.att_head_dim * self.num_heads)
        self.key_layer = torch.nn.Linear(att_input_dim, self.att_head_dim * self.num_heads)
        self.value_layer = torch.nn.Linear(att_input_dim, self.att_head_dim * self.num_heads)
    #
    def forward(self, ts10_input: torch.Tensor, mask=False):
        """
        ts10_input: shape (batch_size, seq_size, att_input_dim)
        Return:
            ts05C_ctx: shape (batch_size, seq_size, att_output_dim)
        """
        batch_size=ts10_input.shape[0]
        ### get query matrix
        ### shape (batch_size, num_heads, seq_size, att_head_dim)
        ts03A_query = (self.query_layer(ts10_input)
            .reshape(batch_size, -1, self.num_heads, self.att_head_dim)
            .permute(0,2,1,3))
        ### get key matrix
        ### shape (batch_size, num_heads, seq_size, att_head_dim)
        ts03D_key = (self.key_layer(ts10_input)
            .reshape(batch_size, -1, self.num_heads, self.att_head_dim)
            .permute(0,2,1,3))
        ### get value matrix
        ts03F_value = (self.value_layer(ts10_input)
            .reshape(batch_size, -1, self.num_heads, self.att_head_dim)
            .permute(0,2,1,3))
        ## use layer_norm in similarity matrix to prevent exploding
        ts05A = (torch.matmul(ts03A_query, ts03D_key.transpose(2,3))
            / torch.sqrt(torch.tensor(self.att_head_dim))
            )
        if mask:
            ts02A_neg_inf = torch.full(ts05A.shape, -np.inf)
            ts02D_mask = torch.triu(ts02A_neg_inf, diagonal=1)
            ts05A = ts05A + ts02D_mask
        #
        ts05B_score = torch.nn.functional.softmax(ts05A, dim=-1)
        ts05C_ctx = (torch.matmul(ts05B_score, ts03F_value) ## shape (batch_size, num_head, seq_size, self.att_head_dim)
            .permute(0,2,1,3) ## shape (batch_size, seq_size, num_head, self.att_head_dim)
            .reshape(batch_size, -1, self.num_heads * self.att_head_dim)
            )
        ###
        return ts05C_ctx
#
class BlockSelfAtt(torch.nn.Module):
    # pylint: disable=too-many-instance-attributes
    # pylint: disable=too-few-public-methods
    """
    There are two sub-blocks: attention and ffw
    There is layernorm before each sub-block, and residual layer for each sub-block
    """
    def __init__(self, att_input_dim: int, att_output_dim: int,
            ffw_dim:int, num_heads: int):
        #
        super().__init__()
        ### attention
        self.att_layer_norm = torch.nn.LayerNorm(att_input_dim)
        self.self_att = MultiHeadSelfAtt(att_input_dim, att_output_dim, num_heads)
        self.att_residual_layer = torch.nn.Linear(att_input_dim, att_output_dim)
        ### FFW
        self.ffw_layer_norm = torch.nn.LayerNorm(att_output_dim)
        self.ffw_layer = torch.nn.Sequential(
            torch.nn.Linear(att_output_dim, ffw_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(ffw_dim, att_output_dim)
            )
        self.ffw_residual_layer = torch.nn.Linear(att_output_dim, att_output_dim)
    #
    def forward(self, ts10_input: torch.Tensor, mask=True):
        """
        ts10_input: shape (batch_size, seq_size, att_input_dim)
        Return:
            ts05C_ctx: shape (batch_size, seq_size, att_output_dim)
        """
        batch_size=ts10_input.shape[0]
        #
        ts01_layer_norm = self.att_layer_norm(ts10_input)
        ts04_self_att = self.self_att(ts01_layer_norm, mask)
        ## shape (batch_size, seq_size, att_output_dim)
        ts07A_resi = self.att_residual_layer(ts10_input)
        ts07B_resi = ts07A_resi + ts04_self_att
        ###
        ts15_layer_norm = self.ffw_layer_norm(ts07B_resi)
        ts17_ffw = self.ffw_layer(ts15_layer_norm)
        ts19_resi = self.ffw_residual_layer(ts07B_resi) + ts17_ffw
        ###
        return ts19_resi
#
class CNNCharEncode(torch.nn.Module):
    def __init__(self, char_dim: int, width_list, num_filters_list):
        """
        char_dim: embedding size of a character
        width_list: list of width sizes for convolution
        num_filters_list: number of filters corresponds to width_list
        for example: width_list = [1,2,3], num_filters_list=[128,128,512]
            means 128 filters of size 1*char_dim, 128 filters of size 2*char_dim,
            512 filters of size 3*char_dim
        """
        #
        super().__init__()
        assert len(width_list)==len(num_filters_list)
        ### CNN
        cnn_list = []
        for w, h in zip(width_list, num_filters_list):
            cnn_list.append(torch.nn.Conv1d(char_dim, h, w, stride=1,groups=1))
        #
        self.cnn_list = torch.nn.ModuleList(cnn_list)
        ### Highway
        self.out_dim = sum(num_filters_list)
        self.wh = torch.nn.Linear(self.out_dim, self.out_dim)
        self.wt = torch.nn.Linear(self.out_dim, self.out_dim)
    #
    def forward(self, ts10_input: torch.Tensor):
        """
        ts10_input: shape (batch_size, seq_size, max_len_word+2, char_dim)
        2 is begin and end token of a word
        ts16_out: shape (batch_size, seq_size, out_dim)
        """
        batch_size, seq_size, max_len_word, char_dim = ts10_input.shape
        ### need to compress batch_size and seq_size
        ts11A = ts10_input.reshape(-1, max_len_word, char_dim)
        ### need to transpose to use CNN
        ts11A = ts11A.permute(0,2,1)
        #
        ts13_out = []
        for tcnn in self.cnn_list:
            tanh = torch.tanh(tcnn(ts11A))
            max_over_time = torch.max(tanh, dim=-1)[0]
            ts13_out.append(max_over_time)
        #
        ts13_out = torch.cat(ts13_out, dim=-1)
        ### highway
        ts15_wh = torch.nn.functional.relu(self.wh(ts13_out))
        ts15_wt = torch.sigmoid(self.wt(ts13_out))
        ts16_out = ts15_wt*ts15_wh + (1-ts15_wt)*ts13_out
        ts16_out = ts16_out.reshape(batch_size, seq_size, -1)
        return ts16_out
#
class TwoTowers(torch.nn.Module):
    def __init__(self, num_total_char, char_dim, width_list, num_filters_list,
            att_output_dim, ffw_dim, num_heads, num_block):
        """
        one forward tower and one backward tower
        """
        #
        super().__init__()
        self.char_emb_layer = torch.nn.Embedding(num_total_char, char_dim)
        self.cnn_char_layer = CNNCharEncode(char_dim, width_list, num_filters_list)
        self.cnn_out_dim = self.cnn_char_layer.out_dim
        ### forward tower
        fw_block = [BlockSelfAtt(self.cnn_out_dim, att_output_dim, ffw_dim, num_heads)]
        for _ in range(num_block-1):
            fw_block.append(BlockSelfAtt(att_output_dim, att_output_dim, ffw_dim, num_heads))
        #
        self.fw_block = torch.nn.ModuleList(fw_block)
        ### backward tower
        bw_block = [BlockSelfAtt(self.cnn_out_dim, att_output_dim, ffw_dim, num_heads)]
        for _ in range(num_block-1):
            bw_block.append(BlockSelfAtt(att_output_dim, att_output_dim, ffw_dim, num_heads))
        #
        self.bw_block = torch.nn.ModuleList(bw_block)
    #
    def forward(self, ts10A_input_fw, ts20A_input_bw, start_pos_fw, start_pos_bw):
        """
        ts10A_input_fw, ts10B_input_bw: shape (batch_size, seq_size, max_len_word+2)
        """
        end_pos_fw = start_pos_fw+ts10A_input_fw.shape[1]-1
        end_pos_bw = start_pos_bw+ts20A_input_bw.shape[1]-1
        ### forward tower
        ts11_fw_char_emb = self.char_emb_layer(ts10A_input_fw)
        ts12_fw_cnn = self.cnn_char_layer(ts11_fw_char_emb)
        ts13_fw_pe = positional_embedding(self.cnn_out_dim, start_pos_fw, end_pos_fw)
        ts14_fw_pe = ts12_fw_cnn+ts13_fw_pe
        ts15_fw_att = self.fw_block[0](ts14_fw_pe, mask=True)
        for fw_att in self.fw_block[1:]:
            ts15_fw_att = fw_att(ts15_fw_att)
        ### backward tower
        ts21_bw_char_emb = self.char_emb_layer(ts20A_input_bw)
        ts22_bw_cnn = self.cnn_char_layer(ts21_bw_char_emb)
        ts23_bw_pe = positional_embedding(self.cnn_out_dim, start_pos_bw, end_pos_bw)
        ts24_bw_pe = ts22_bw_cnn+ts23_bw_pe
        ts24_bw_pe = torch.flip(ts24_bw_pe, (1,)) ### revert sequence order
        ts25_bw_att = self.bw_block[0](ts24_bw_pe, mask=True)
        for bw_att in self.bw_block[1:]:
            ts25_bw_att = bw_att(ts25_bw_att)
        #
        ts25_bw_att = torch.flip(ts25_bw_att, (1,)) ### revert sequence to origin
        return ts15_fw_att, ts25_bw_att
#
class Pretraining(torch.nn.Module):
    def __init__(self, num_total_char, char_dim, width_list, num_filters_list,
            att_output_dim, ffw_dim, num_heads, num_block, num_heads_last,
            vocab_size, cutoffs):
        """
        """
        #
        super().__init__()
        self.two_tower = TwoTowers(num_total_char, char_dim, width_list, num_filters_list,
            att_output_dim, ffw_dim, num_heads, num_block)
        self.last_att = MultiHeadAtt(2*att_output_dim, att_output_dim,
            att_output_dim, att_output_dim, num_heads_last)
        self.ffw_layer = torch.nn.Sequential(
            torch.nn.Linear(att_output_dim, ffw_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(ffw_dim, att_output_dim)
            )
        self.adapt_softmax = torch.nn.AdaptiveLogSoftmaxWithLoss(att_output_dim, vocab_size, cutoffs)
    #
    def forward(self, ts10A_input_fw, ts20A_input_bw, start_pos_fw, start_pos_bw, target=None):
        ts30A_fw, ts30D_bw = self.two_tower(ts10A_input_fw,ts20A_input_bw,start_pos_fw,start_pos_bw)
        ts31_q = torch.cat((ts30A_fw[:,-1,:], ts30D_bw[:,0,:]), dim=-1).unsqueeze(1)
        ts32_k = ts33_v = torch.cat((ts30A_fw,ts30D_bw), dim=1)
        ts35_att = self.last_att(ts31_q, ts32_k, ts33_v, mask=False)
        ts36_ffw = self.ffw_layer(ts35_att).squeeze(1)
        ts40_loss = ts41_pred = None
        if target is None:
            ts41_pred = self.adapt_softmax.predict(ts36_ffw)
        else:
            _, ts40_loss = self.adapt_softmax(ts36_ffw, target)
        #
        return ts40_loss, ts41_pred
#
class Finetuning(torch.nn.Module):
    def __init__(self, num_total_char, char_dim, width_list, num_filters_list,
            att_output_dim, ffw_dim, num_heads, num_block,
            lstm_hid_dim, lstm_proj_dim, num_tags, pretrain_twotower):
        """
        """
        #
        super().__init__()
        self.lstm_hid_dim = lstm_hid_dim
        self.lstm_proj_dim = lstm_proj_dim
        #
        if pretrain_twotower is not None:
            self.two_tower = pretrain_twotower
        else:
            self.two_tower = TwoTowers(num_total_char, char_dim, width_list, num_filters_list,
                att_output_dim, ffw_dim, num_heads, num_block)
        #
        self.ffw_layer = torch.nn.Linear(2*att_output_dim, att_output_dim)
        self.bilstm = torch.nn.LSTM(att_output_dim, lstm_hid_dim,
            batch_first=True, bidirectional=True, proj_size=lstm_proj_dim)
        self.lstm_resi = torch.nn.Linear(att_output_dim, 2*lstm_proj_dim)
        self.hidden2tags = torch.nn.Linear(2*lstm_proj_dim, num_tags)
        self.crf = CRF(num_tags)
    #
    def forward(self, ts10A_input, target=None):
        ts30A, ts30B = self.two_tower(ts10A_input, ts10A_input, 0, 0)
        ts31_comb = torch.cat((ts30A,ts30B), dim=-1)
        ts32_ffw = self.ffw_layer(ts31_comb)
        batch_size = ts32_ffw.shape[0]
        ts33D_init_state = (torch.randn(2, batch_size, self.lstm_proj_dim),
            torch.randn(2, batch_size, self.lstm_hid_dim))
        ts33_bilstm, _ = self.bilstm(ts32_ffw, ts33D_init_state)
        ts34_resi = self.lstm_resi(ts32_ffw)+ts33_bilstm
        ts35_emission = self.hidden2tags(ts34_resi)
        ts36_loss = ts37_pred = None
        if target is not None:
            ts36_loss = self.crf(ts35_emission, target)
        else:
            ts37_pred = self.crf.decode(ts35_emission)
        #
        return ts36_loss, ts37_pred
#
