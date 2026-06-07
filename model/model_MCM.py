import torch
import torch.nn as nn
import torch.nn.functional as F

#**********************************************************************************
# Copyright (c) 2023 Process Systems Engineering (AVT.SVT), RWTH Aachen University
#
# This program and the accompanying materials are made available under the
# terms of the Eclipse Public License 2.0 which is available at
# http://www.eclipse.org/legal/epl-2.0.
#
# SPDX-License-Identifier: EPL-2.0
#
# The source code can be found here:
# https://git.rwth-aachen.de/avt-svt/public/GDI-NN
#
# Notes:
# - This model is based on the paper by Chen, G., Song, Z., Qi, Z., & Sundmacher, K. (2021). Neural recommender system for the activity coefficient prediction and UNIFAC model extension of ionic liquid‐solute systems. AIChE Journal, 67(4), e17171.
# - This code was adpated from the reimplementation by Rittig, J. G., Hicham, K. B., Schweidtmann, A. M., Dahmen, M., & Mitsos, A. (2023). Graph neural networks for temperature-dependent activity coefficient prediction of solutes in ionic liquids. Computers & Chemical Engineering, 171, 108153.
#
#*********************************************************************************


def get_activation(activation, get_nn=False):
    if (activation == None) or (activation in ["relu", "ReLU", "RELU"]):
        if get_nn: return nn.ReLU
        return F.relu
    elif activation in ["elu", "ELU"]:
        if get_nn: return nn.ELU
        return F.elu
    elif activation in ["LeakyReLU", "LeakyRELU", "leakyReLU", "leakyrelu", "leakyRELU", "leaky_relu", "Leaky_ReLU", "Leaky_RELU"]:
        if get_nn: return nn.LeakyReLU
        return F.leaky_relu
    elif activation in ["sigmoid", "Sigmoid", "SIGMOID"]:
        if get_nn: return nn.Sigmoid
        return F.sigmoid
    elif activation in ["softplus", "Softplus", "SOFTPLUS"]:
        if get_nn: return nn.Softplus
        return F.softplus
    elif activation in ["silu", "SiLU", "SILU"]:
        if get_nn: return nn.SiLU
        return F.silu

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_mlp_module(dim_in, dim_hidden, dropout):
    mlp_module_list = nn.ModuleList()
    mlp_module_list.append(
        nn.Sequential(
            nn.Embedding(dim_in, dim_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
        )
    )
    return mlp_module_list

class MCM_multiMLP(nn.Module):

    def __init__(self, solvent_id_max, dim_hidden_channels=128, dropout_hidden=0.05, dropout_interaction=0.03, mlp_activation=None, mlp_num_hid_layers=1, **kwargs):
        super().__init__()
        
        self.mlp_activation = get_activation(mlp_activation, get_nn=True)

        self.dropout_p1 = dropout_hidden
        self.dropout_p2 = dropout_interaction
    
        self.dim_hidden_channels = dim_hidden_channels

        self.solvent_emb = get_mlp_module(solvent_id_max + 1, self.dim_hidden_channels, self.dropout_p1)
        mid_emb = 2 * self.dim_hidden_channels        

        list_layers_end_1 = [
            nn.Linear(mid_emb + 2, mid_emb),
            self.mlp_activation()
        ]
        if mlp_num_hid_layers > 1:
            for _ in range(mlp_num_hid_layers-1):
                list_layers_end_1.append(
                    nn.Linear(mid_emb, mid_emb),
                )
                list_layers_end_1.append(
                    self.mlp_activation()
                )
        list_layers_end_1.append(
            nn.Linear(mid_emb, 1)
        )
        list_layers_end_2 = [
            nn.Linear(mid_emb + 2, mid_emb),
            self.mlp_activation()
        ]
        if mlp_num_hid_layers > 1:
            for _ in range(mlp_num_hid_layers-1):
                list_layers_end_2.append(
                    nn.Linear(mid_emb, mid_emb),
                )
                list_layers_end_2.append(
                    self.mlp_activation()
                )
        list_layers_end_2.append(
            nn.Linear(mid_emb, 1)
        )
        self.layers_end = nn.ModuleList([
            nn.Sequential(*list_layers_end_1),
            nn.Sequential(*list_layers_end_2)
        ])


    def forward(self, solvdata, empty_solvsys, gamma_grad=False):
        '''
          Forward pass
        '''
        solv1x = solvdata["solv1_x"].cpu()
        solv1x.requires_grad = True
        data_dict = {
            'solvent':   solvdata['solv1_id'].cpu(),
            'solute':   solvdata['solv2_id'].cpu(),
        }

        x_solvent = self.solvent_emb[0](data_dict['solvent'])
        x_solute = self.solvent_emb[0](data_dict['solute'])
        
        h = torch.cat([x_solvent, solv1x[:,None], x_solute, 1-solv1x[:,None]], dim=1).to(torch.float32)

        output_y1 = self.layers_end[0](h)
        output_y2 = self.layers_end[1](h)
        output = torch.cat([output_y1, output_y2], dim=1)

        if gamma_grad:
            y1_x1 = torch.autograd.grad(output[:,0].sum(), solv1x, create_graph=True)[0] 
            y2_x1 = torch.autograd.grad(output[:,1].sum(), solv1x, create_graph=True)[0]
            return output, y1_x1, y2_x1   

        return output

