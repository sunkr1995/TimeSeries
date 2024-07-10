from enum import Enum
from typing import List, NewType, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

ACTIVATIONS = [
    "ReLU",
    "RReLU",
    "PReLU",
    "ELU",
    "Softplus",
    "Tanh",
    "SELU",
    "LeakyReLU",
    "Sigmoid",
    "GELU",
]


class _GType(Enum):
    GENERIC = 1
    TREND = 2
    SEASONALITY = 3

def raise_if_not(
    condition: bool,
    message: str = "",
):
    """
    Checks provided boolean condition and raises a ValueError if it evaluates to False.
    It logs the error to the provided logger before raising it.

    Parameters
    ----------
    condition
        The boolean condition to be checked.
    message
        The message of the ValueError.

    Raises
    ------
    ValueError
        if `condition` is not satisfied
    """

    if not condition:
        raise ValueError(message)



GTypes = NewType("GTypes", _GType)

class MonteCarloDropout(nn.Dropout):
    """
    Defines Monte Carlo dropout Module as defined
    in the paper https://arxiv.org/pdf/1506.02142.pdf.
    In summary, This technique uses the regular dropout
    which can be interpreted as a Bayesian approximation of
    a well-known probabilistic model: the Gaussian process.
    We can treat the many different networks
    (with different neurons dropped out) as Monte Carlo samples
    from the space of all available models. This provides mathematical
    grounds to reason about the model’s uncertainty and, as it turns out,
    often improves its performance.
    """

    # mc dropout is deactivated at init; see `MonteCarloDropout.mc_dropout_enabled` for more info
    _mc_dropout_enabled = False

    def forward(self, input: Tensor) -> Tensor:
        # NOTE: we could use the following line in case a different rate
        # is used for inference:
        # return F.dropout(input, self.applied_rate, True, self.inplace)
        return F.dropout(input, self.p, self.mc_dropout_enabled, self.inplace)

    @property
    def mc_dropout_enabled(self) -> bool:
        # mc dropout is only activated on `PLForecastingModule.on_predict_start()`
        # otherwise, it is activated based on the `model.training` flag.
        return self._mc_dropout_enabled or self.training


class _TrendGenerator(nn.Module):
    def __init__(self, expansion_coefficient_dim, target_length):
        super().__init__()

        # basis is of size (expansion_coefficient_dim, target_length)
        basis = torch.stack(
            [
                (torch.arange(target_length) / target_length) ** i
                for i in range(expansion_coefficient_dim)
            ],
            dim=1,
        ).T

        self.basis = nn.Parameter(basis, requires_grad=False)

    def forward(self, x):
        return torch.matmul(x, self.basis)


class _SeasonalityGenerator(nn.Module):
    def __init__(self, target_length):
        super().__init__()
        half_minus_one = int(target_length / 2 - 1)
        cos_vectors = [
            torch.cos(torch.arange(target_length) / target_length * 2 * np.pi * i)
            for i in range(1, half_minus_one + 1)
        ]
        sin_vectors = [
            torch.sin(torch.arange(target_length) / target_length * 2 * np.pi * i)
            for i in range(1, half_minus_one + 1)
        ]

        # basis is of size (2 * int(target_length / 2 - 1) + 1, target_length)
        basis = torch.stack(
            [torch.ones(target_length)] + cos_vectors + sin_vectors, dim=1
        ).T

        self.basis = nn.Parameter(basis, requires_grad=False)

    def forward(self, x):
        return torch.matmul(x, self.basis)


class _Block(nn.Module):
    def __init__(
        self,
        num_layers: int,
        layer_width: int,
        nr_params: int,
        expansion_coefficient_dim: int,
        input_chunk_length: int,
        target_length: int,
        g_type: GTypes,
        batch_norm: bool,
        dropout: float,
        activation: str,
    ):
        """PyTorch module implementing the basic building block of the N-BEATS architecture.

        The blocks produce outputs of size (target_length, nr_params); i.e.
        "one vector per parameter". The parameters are predicted only for forecast outputs.
        Backcast outputs are in the original "domain".

        Parameters
        ----------
        num_layers
            The number of fully connected layers preceding the final forking layers.
        layer_width
            The number of neurons that make up each fully connected layer.
        nr_params
            The number of parameters of the likelihood (or 1 if no likelihood is used)
        expansion_coefficient_dim
            The dimensionality of the waveform generator parameters, also known as expansion coefficients.
            Used in the generic architecture and the trend module of the interpretable architecture, where it determines
            the degree of the polynomial basis.
        input_chunk_length
            The length of the input sequence fed to the model.
        target_length
            The length of the forecast of the model.
        g_type
            The type of function that is implemented by the waveform generator.
        batch_norm
            Whether to use batch norm
        dropout
            Dropout probability
        activation
            The activation function of encoder/decoder intermediate layer.

        Inputs
        ------
        x of shape `(batch_size, input_chunk_length)`
            Tensor containing the input sequence.

        Outputs
        -------
        x_hat of shape `(batch_size, input_chunk_length)`
            Tensor containing the 'backcast' of the block, which represents an approximation of `x`
            given the constraints of the functional space determined by `g`.
        y_hat of shape `(batch_size, output_chunk_length)`
            Tensor containing the forward forecast of the block.

        """
        super().__init__()
        
        self.num_layers = num_layers
        self.layer_width = layer_width
        self.target_length = target_length
        self.nr_params = nr_params
        self.g_type = g_type
        self.dropout = dropout
        self.batch_norm = batch_norm

        raise_if_not(
            activation in ACTIVATIONS, f"'{activation}' is not in {ACTIVATIONS}"
        )
        self.activation = getattr(nn, activation)()

        # fully connected stack before fork
        self.linear_layer_stack_list = [nn.Linear(input_chunk_length, layer_width)]
        for _ in range(num_layers - 1):
            self.linear_layer_stack_list.append(nn.Linear(layer_width, layer_width))

            if self.batch_norm:
                self.linear_layer_stack_list.append(
                    nn.BatchNorm1d(num_features=self.layer_width)
                )

            if self.dropout > 0:
                self.linear_layer_stack_list.append(MonteCarloDropout(p=self.dropout))

        self.fc_stack = nn.ModuleList(self.linear_layer_stack_list)

        # Fully connected layer producing forecast/backcast expansion coefficients (waveform generator parameters).
        # The coefficients are emitted for each parameter of the likelihood.
        if g_type == _GType.SEASONALITY:
            self.backcast_linear_layer = nn.Linear(
                layer_width, 2 * int(input_chunk_length / 2 - 1) + 1
            )
            self.forecast_linear_layer = nn.Linear(
                layer_width, nr_params * (2 * int(target_length / 2 - 1) + 1)
            )
        else:
            self.backcast_linear_layer = nn.Linear(
                layer_width, expansion_coefficient_dim
            )
            self.forecast_linear_layer = nn.Linear(
                layer_width, nr_params * expansion_coefficient_dim
            )

        # waveform generator functions
        if g_type == _GType.GENERIC:
            self.backcast_g = nn.Linear(expansion_coefficient_dim, input_chunk_length)
            self.forecast_g = nn.Linear(expansion_coefficient_dim, target_length)
        elif g_type == _GType.TREND:
            self.backcast_g = _TrendGenerator(
                expansion_coefficient_dim, input_chunk_length
            )
            self.forecast_g = _TrendGenerator(expansion_coefficient_dim, target_length)
        elif g_type == _GType.SEASONALITY:
            self.backcast_g = _SeasonalityGenerator(input_chunk_length)
            self.forecast_g = _SeasonalityGenerator(target_length)
        else:
            raise ValueError("g_type not supported")

    def forward(self, x):
        batch_size = x.shape[0]

        # fully connected layer stack
        for layer in self.linear_layer_stack_list:
            x = self.activation(layer(x))

        # forked linear layers producing waveform generator parameters
        theta_backcast = self.backcast_linear_layer(x)
        theta_forecast = self.forecast_linear_layer(x)

        # set the expansion coefs in last dimension for the forecasts
        theta_forecast = theta_forecast.view(batch_size, self.nr_params, -1)

        # waveform generator applications (project the expansion coefs onto basis vectors)
        x_hat = self.backcast_g(theta_backcast)
        y_hat = self.forecast_g(theta_forecast)

        # Set the distribution parameters as the last dimension
        y_hat = y_hat.reshape(x.shape[0], self.target_length, self.nr_params)

        return x_hat, y_hat


class _Stack(nn.Module):
    def __init__(
        self,
        num_blocks: int,
        num_layers: int,
        layer_width: int,
        nr_params: int,
        expansion_coefficient_dim: int,
        input_chunk_length: int,
        target_length: int,
        g_type: GTypes,
        batch_norm: bool,
        dropout: float,
        activation: str,
    ):
        """PyTorch module implementing one stack of the N-BEATS architecture that comprises multiple basic blocks.

        Parameters
        ----------
        num_blocks
            The number of blocks making up this stack.
        num_layers
            The number of fully connected layers preceding the final forking layers in each block.
        layer_width
            The number of neurons that make up each fully connected layer in each block.
        nr_params
            The number of parameters of the likelihood (or 1 if no likelihood is used)
        expansion_coefficient_dim
            The dimensionality of the waveform generator parameters, also known as expansion coefficients.
        input_chunk_length
            The length of the input sequence fed to the model.
        target_length
            The length of the forecast of the model.
        g_type
            The function that is implemented by the waveform generators in each block.
        batch_norm
            whether to apply batch norm on first block of this stack
        dropout
            Dropout probability
        activation
            The activation function of encoder/decoder intermediate layer.

        Inputs
        ------
        stack_input of shape `(batch_size, input_chunk_length)`
            Tensor containing the input sequence.

        Outputs
        -------
        stack_residual of shape `(batch_size, input_chunk_length)`
            Tensor containing the 'backcast' of the block, which represents an approximation of `x`
            given the constraints of the functional space determined by `g`.
        stack_forecast of shape `(batch_size, output_chunk_length)`
            Tensor containing the forward forecast of the stack.

        """
        super().__init__()

        self.input_chunk_length = input_chunk_length
        self.target_length = target_length
        self.nr_params = nr_params
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.activation = activation

        if g_type == _GType.GENERIC:
            self.blocks_list = [
                _Block(
                    num_layers,
                    layer_width,
                    nr_params,
                    expansion_coefficient_dim,
                    input_chunk_length,
                    target_length,
                    g_type,
                    batch_norm=(
                        self.batch_norm and i == 0
                    ),  # batch norm only on first block of first stack
                    dropout=self.dropout,
                    activation=self.activation,
                )
                for i in range(num_blocks)
            ]
        else:
            # same block instance is used for weight sharing
            interpretable_block = _Block(
                num_layers,
                layer_width,
                nr_params,
                expansion_coefficient_dim,
                input_chunk_length,
                target_length,
                g_type,
                batch_norm=self.batch_norm,
                dropout=self.dropout,
                activation=self.activation,
            )
            self.blocks_list = [interpretable_block] * num_blocks

        self.blocks = nn.ModuleList(self.blocks_list)

    def forward(self, x):
        # One forecast vector per parameter in the distribution
        stack_forecast = torch.zeros(
            x.shape[0],
            self.target_length,
            self.nr_params,
            device=x.device,
            dtype=x.dtype,
        )

        for block in self.blocks_list:
            # pass input through block
            x_hat, y_hat = block(x)

            # add block forecast to stack forecast
            stack_forecast = stack_forecast + y_hat

            # subtract backcast from input to produce residual
            x = x - x_hat

        stack_residual = x

        return stack_residual, stack_forecast
    

class NBEATSModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        input_chunk_length:int,
        output_dim: int,
        output_chunk_length:int,
        generic_architecture: bool,
        num_stacks: int,
        num_blocks: int,
        num_layers: int,
        layer_widths: Union[int, List[int]] = 256,
        expansion_coefficient_dim: int = 5,
        trend_polynomial_degree: int = 2,
        batch_norm: bool = False,
        dropout: float = 0.0,
        activation: str = "ReLU"
    ):
        """PyTorch module implementing the N-BEATS architecture.

        Parameters
        ----------
        output_dim
            Number of output components in the target
        generic_architecture
            Boolean value indicating whether the generic architecture of N-BEATS is used.
            If not, the interpretable architecture outlined in the paper (consisting of one trend
            and one seasonality stack with appropriate waveform generator functions).
        num_stacks
            The number of stacks that make up the whole model. Only used if `generic_architecture` is set to `True`.
        num_blocks
            The number of blocks making up every stack.
        num_layers
            The number of fully connected layers preceding the final forking layers in each block of every stack.
        layer_widths
            Determines the number of neurons that make up each fully connected layer in each block of every stack.
            If a list is passed, it must have a length equal to `num_stacks` and every entry in that list corresponds
            to the layer width of the corresponding stack. If an integer is passed, every stack will have blocks
            with FC layers of the same width.
        expansion_coefficient_dim
            The dimensionality of the waveform generator parameters, also known as expansion coefficients.
            Only used if `generic_architecture` is set to `True`.
        trend_polynomial_degree
            The degree of the polynomial used as waveform generator in trend stacks. Only used if
            `generic_architecture` is set to `False`.
        batch_norm
            Whether to apply batch norm on first block of the first stack
        dropout
            Dropout probability
        activation
            The activation function of encoder/decoder intermediate layer.

        Inputs
        ------
        x of shape `(batch_size, input_chunk_length)`
            Tensor containing the input sequence.

        Outputs
        -------
        y of shape `(batch_size, output_chunk_length, target_size/output_dim, nr_params)`
            Tensor containing the output of the NBEATS module.

        """
        super(NBEATSModel, self).__init__()

        nr_params = 1

        self.input_dim = input_dim
        self.input_chunk_length = input_chunk_length
        self.output_dim = output_dim
        self.output_chunk_length = output_chunk_length
        self.nr_params = nr_params
        self.input_chunk_length_multi = self.input_chunk_length * input_dim
        self.target_length = self.output_chunk_length * input_dim
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.activation = activation

        if isinstance(layer_widths, int):
            layer_widths = [layer_widths] * num_stacks

        if generic_architecture:
            self.stacks_list = [
                _Stack(
                    num_blocks,
                    num_layers,
                    layer_widths[i],
                    nr_params,
                    expansion_coefficient_dim,
                    self.input_chunk_length_multi,
                    self.target_length,
                    _GType.GENERIC,
                    batch_norm=(
                        self.batch_norm and i == 0
                    ),  # batch norm only on first block of first stack
                    dropout=self.dropout,
                    activation=self.activation,
                )
                for i in range(num_stacks)
            ]
        else:
            num_stacks = 2
            trend_stack = _Stack(
                num_blocks,
                num_layers,
                layer_widths[0],
                nr_params,
                trend_polynomial_degree + 1,
                self.input_chunk_length_multi,
                self.target_length,
                _GType.TREND,
                batch_norm=self.batch_norm,
                dropout=self.dropout,
                activation=self.activation,
            )
            seasonality_stack = _Stack(
                num_blocks,
                num_layers,
                layer_widths[1],
                nr_params,
                -1,
                self.input_chunk_length_multi,
                self.target_length,
                _GType.SEASONALITY,
                batch_norm=self.batch_norm,
                dropout=self.dropout,
                activation=self.activation,
            )
            self.stacks_list = [trend_stack, seasonality_stack]

        self.num_stacks = num_stacks
        self.num_blocks = num_blocks
        self.num_layers = num_layers
        self.layer_widths = layer_widths
        self.stacks = nn.ModuleList(self.stacks_list)

        # setting the last backcast "branch" to be not trainable (without next block/stack, it doesn't need to be
        # backpropagated). Removing this lines would cause logtensorboard to crash, since no gradient is stored
        # on this params (the last block backcast is not part of the final output of the net).
        self.stacks_list[-1].blocks[-1].backcast_linear_layer.requires_grad_(False)
        self.stacks_list[-1].blocks[-1].backcast_g.requires_grad_(False)

    def forward(self, x):
        # if x1, x2,... y1, y2... is one multivariate ts containing x and y, and a1, a2... one covariate ts
        # we reshape into x1, y1, a1, x2, y2, a2... etc
        x = torch.reshape(x, (x.shape[0], self.input_chunk_length_multi, 1))
        # squeeze last dimension (because model is univariate)
        x = x.squeeze(dim=2)

        # One vector of length target_length per parameter in the distribution
        y = torch.zeros(
            x.shape[0],
            self.target_length,
            self.nr_params,
            device=x.device,
            dtype=x.dtype,
        )

        for stack in self.stacks_list:
            # compute stack output
            stack_residual, stack_forecast = stack(x)

            # add stack forecast to final output
            y = y + stack_forecast

            # set current stack residual as input for next stack
            x = stack_residual

        # In multivariate case, we get a result [x1_param1, x1_param2], [y1_param1, y1_param2], [x2..], [y2..], ...
        # We want to reshape to original format. We also get rid of the covariates and keep only the target dimensions.
        # The covariates are by construction added as extra time series on the right side. So we need to get rid of this
        # right output (keeping only :self.output_dim).
        y = y.view(
            y.shape[0], self.output_chunk_length, self.input_dim
        )[:, :, : self.output_dim]

        return y
