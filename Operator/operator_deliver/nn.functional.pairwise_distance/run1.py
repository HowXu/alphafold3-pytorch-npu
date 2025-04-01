"""
原始函数定义
def forward(
        self,
        pred_coords: Float['b n 3'],
        true_coords: Float['b n 3'],
        pred_frames: Float['b n 3 3'],
        true_frames: Float['b n 3 3'],
        mask: Bool['b n'] | None = None,
    ) -> Float['b n n']:
        
        pred_coords: predicted coordinates
        true_coords: true coordinates
        pred_frames: predicted frames
        true_frames: true frames
        
        # to pairs

        seq = pred_coords.shape[1]
        
        pair2seq = partial(rearrange, pattern='b n m ... -> b (n m) ...')
        seq2pair = partial(rearrange, pattern='b (n m) ... -> b n m ...', n = seq, m = seq)
        
        pair_pred_coords = pair2seq(repeat(pred_coords, 'b n d -> b n m d', m = seq))
        pair_true_coords = pair2seq(repeat(true_coords, 'b n d -> b n m d', m = seq))
        pair_pred_frames = pair2seq(repeat(pred_frames, 'b n d e -> b m n d e', m = seq))
        pair_true_frames = pair2seq(repeat(true_frames, 'b n d e -> b m n d e', m = seq))
        
        # Express predicted coordinates in predicted frames
        pred_coords_transformed = self.express_coordinates_in_frame(pair_pred_coords, pair_pred_frames)

        # Express true coordinates in true frames
        true_coords_transformed = self.express_coordinates_in_frame(pair_true_coords, pair_true_frames)

        # Compute alignment errors
        alignment_errors = F.pairwise_distance(pred_coords_transformed, true_coords_transformed, eps = self.eps)
"""
from functools import partial

from torch.nn.functional import pairwise_distance
from einops import rearrange, repeat
import torch
from alphafold3_pytorch import ExpressCoordinatesInFrame
import torch_npu

if torch.npu.is_available():
    device = torch.device("npu")
    print("NPU is available!")
else:
    device = torch.device("cpu")
    print("NPU is NOT available, falling back to CPU.")

b = 4096
n = 50
pred_coords = torch.randn(b, n, 3, dtype=torch.float32).to(device)
true_coords = torch.randn(b, n, 3, dtype=torch.float32).to(device)
pred_frames = torch.randn(b, n, 3, 3, dtype=torch.float32).to(device)
true_frames = torch.randn(b, n, 3, 3, dtype=torch.float32).to(device)
# 随机数据取代输入参数
seq = pred_coords.shape[1]
pair2seq = partial(rearrange, pattern='b n m ... -> b (n m) ...')
pair_pred_coords = pair2seq(repeat(pred_coords, 'b n d -> b n m d', m = seq))
pair_true_coords = pair2seq(repeat(true_coords, 'b n d -> b n m d', m = seq))
pair_pred_frames = pair2seq(repeat(pred_frames, 'b n d e -> b m n d e', m = seq))
pair_true_frames = pair2seq(repeat(true_frames, 'b n d e -> b m n d e', m = seq))

# 我看不懂这个写法 只有forward函数调用但是原文里面是直接掉了self的函数，你是做了继承还是组合？
pred_coords_transformed = ExpressCoordinatesInFrame().forward(pair_pred_coords, pair_pred_frames)

        # Express true coordinates in true frames
true_coords_transformed = ExpressCoordinatesInFrame().forward(pair_true_coords, pair_true_frames)

        # Compute alignment errors
alignment_errors = pairwise_distance(pred_coords_transformed, true_coords_transformed, eps = 1e-8) # 这个值我也不知道怎么来的 magic

print("alignment_errors:", alignment_errors)