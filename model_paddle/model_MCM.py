import paddle


def get_activation(activation, get_nn=False):
    if activation == None or activation in ["relu", "ReLU", "RELU"]:
        if get_nn:
            return paddle.nn.ReLU
        return paddle.nn.functional.relu
    elif activation in ["elu", "ELU"]:
        if get_nn:
            return paddle.nn.ELU
        return paddle.nn.functional.elu
    elif activation in [
        "LeakyReLU",
        "LeakyRELU",
        "leakyReLU",
        "leakyrelu",
        "leakyRELU",
        "leaky_relu",
        "Leaky_ReLU",
        "Leaky_RELU",
    ]:
        if get_nn:
            return paddle.nn.LeakyReLU
        return paddle.nn.functional.leaky_relu
    elif activation in ["sigmoid", "Sigmoid", "SIGMOID"]:
        if get_nn:
            return paddle.nn.Sigmoid
        return paddle.nn.functional.sigmoid
    elif activation in ["softplus", "Softplus", "SOFTPLUS"]:
        if get_nn:
            return paddle.nn.Softplus
        return paddle.nn.functional.softplus
    elif activation in ["silu", "SiLU", "SILU"]:
        if get_nn:
            return paddle.nn.SiLU
        return paddle.nn.functional.silu


device = "gpu" if paddle.is_compiled_with_cuda() else "cpu"
paddle.set_device(device)


def get_mlp_module(dim_in, dim_hidden, dropout):
    mlp_module_list = paddle.nn.LayerList()
    mlp_module_list.append(
        paddle.nn.Sequential(
            paddle.nn.Embedding(dim_in, dim_hidden),
            paddle.nn.ReLU(),
            paddle.nn.Dropout(dropout),
            paddle.nn.Linear(dim_hidden, dim_hidden),
            paddle.nn.ReLU(),
            paddle.nn.Dropout(dropout),
            paddle.nn.Linear(dim_hidden, dim_hidden),
            paddle.nn.ReLU(),
            paddle.nn.Dropout(dropout),
            paddle.nn.Linear(dim_hidden, dim_hidden),
            paddle.nn.ReLU(),
        )
    )
    return mlp_module_list


class MCM_multiMLP(paddle.nn.Layer):
    def __init__(
        self,
        solvent_id_max,
        dim_hidden_channels=128,
        dropout_hidden=0.05,
        dropout_interaction=0.03,
        mlp_activation=None,
        mlp_num_hid_layers=1,
        **kwargs
    ):
        super().__init__()
        self.mlp_activation = get_activation(mlp_activation, get_nn=True)
        self.dropout_p1 = dropout_hidden
        self.dropout_p2 = dropout_interaction
        self.dim_hidden_channels = dim_hidden_channels
        self.solvent_emb = get_mlp_module(
            solvent_id_max + 1, self.dim_hidden_channels, self.dropout_p1
        )
        mid_emb = 2 * self.dim_hidden_channels
        list_layers_end_1 = [
            paddle.nn.Linear(mid_emb + 2, mid_emb),
            self.mlp_activation(),
        ]
        if mlp_num_hid_layers > 1:
            for _ in range(mlp_num_hid_layers - 1):
                list_layers_end_1.append(paddle.nn.Linear(mid_emb, mid_emb))
                list_layers_end_1.append(self.mlp_activation())
        list_layers_end_1.append(paddle.nn.Linear(mid_emb, 1))
        list_layers_end_2 = [
            paddle.nn.Linear(mid_emb + 2, mid_emb),
            self.mlp_activation(),
        ]
        if mlp_num_hid_layers > 1:
            for _ in range(mlp_num_hid_layers - 1):
                list_layers_end_2.append(paddle.nn.Linear(mid_emb, mid_emb))
                list_layers_end_2.append(self.mlp_activation())
        list_layers_end_2.append(paddle.nn.Linear(mid_emb, 1))
        self.layers_end = paddle.nn.LayerList(
            [
                paddle.nn.Sequential(*list_layers_end_1),
                paddle.nn.Sequential(*list_layers_end_2),
            ]
        )

    def forward(self, solvdata, empty_solvsys, gamma_grad=False):
        """
        Forward pass
        """
        solv1x = solvdata["solv1_x"].cpu()
        solv1x.stop_gradient = not True
        data_dict = {
            "solvent": solvdata["solv1_id"].cpu(),
            "solute": solvdata["solv2_id"].cpu(),
        }
        x_solvent = self.solvent_emb[0](data_dict["solvent"])
        x_solute = self.solvent_emb[0](data_dict["solute"])
        h = paddle.concat(
            [x_solvent, solv1x[:, None], x_solute, 1 - solv1x[:, None]], dim=1
        ).to(paddle.float32)
        output_y1 = self.layers_end[0](h)
        output_y2 = self.layers_end[1](h)
        output = paddle.concat([output_y1, output_y2], dim=1)
        if gamma_grad:
            y1_x1 = paddle.grad(
                outputs=output[:, 0].sum(), inputs=solv1x, create_graph=True
            )[0]
            y2_x1 = paddle.grad(
                outputs=output[:, 1].sum(), inputs=solv1x, create_graph=True
            )[0]
            return output, y1_x1, y2_x1
        return output
