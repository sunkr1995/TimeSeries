# 关于model config 中的可修改部分
## RNN

```python
{
    "hidden_size": 128,  # 可修改，要求：2的整数幂，大于32
    "num_layers": 3, #  可修改，要求：大于1
    "dropout": 0.5, # 可修改 0.0 / 0.5
    "rnn_type":"gru" #  可修改，lstm 或 gru
}
```

## NBEATS

```python
{
    "generic_architecture": true, # 可修改 true / false
    "num_stacks": 10, # 可修改
    "num_blocks": 1, # 可修改
    "num_layers": 4, # 可修改
    "layer_widths": 256, # 可修改
    "expansion_coefficient_dim": 5, # 可修改
    "trend_polynomial_degree": 2, # 可修改
    "batch_norm": false, # 可修改 true / false
    "dropout": 0.0, #可修改 0.0 / 0.5
    "activation":"ReLU"  # 可修改 ReLU/RReLU/PReLU/ELU/Softplus/Tanh/SELU/LeakyReLU/Sigmoid/GELU
}
```