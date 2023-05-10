import torch
pattern_tensor1: torch.Tensor = torch.tensor([
            [1., 0., 1., 0],
            [-10., 1., -10., 0],
            [-10., -10., 0., 0],
            [-10., 1., -10., 0],
            [1., 0., 1., 0]
        ])
pattern_tensor2: torch.Tensor = torch.tensor([
    [-10., 10., 10., -10.],
    [10., -10., -10., 10.],
    [10., -10., -10., 10.],
    [10., -10., -10., 10.],
    [-10., 10., 10., -10.]
])
pattern_tensor3: torch.Tensor = torch.tensor([
    [-10., 1., -10.,],
    [0., -10., 0.,],
    [-10., 1., -10.,],
])
pattern_tensor4: torch.Tensor = torch.tensor([
    [-10., 0., 0., 0., -10.],
    [-10., 1., 0, 1, -10.],
    [0, 0, -10., 0., 0],
    [-10., 0., 1., 0, -10.],
    [-10., 0., 0., 0., -10]
])
pattern_tensor_dba1: torch.Tensor = torch.tensor([
    [1., 1., 1., 1.],
    [1., 1., 1., 1.],
    [0., -0., 0., 0],
    [-0., 0., -0., 0],
    [0., 0., 0., 0]
])
pattern_tensor_dba2: torch.Tensor = torch.tensor([
    [0., -0., 0., 0],
    [-0., 0., -0., 0],
    [0., 0., 0., 0],
    [1., 1., 1., 1.],
    [1., 1., 1., 1.]
])
pattern_tensor_dba = [pattern_tensor_dba1, pattern_tensor_dba2]
pattern_tensor_normal = [pattern_tensor1, pattern_tensor2, pattern_tensor3, pattern_tensor4]