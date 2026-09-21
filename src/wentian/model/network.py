"""Canonical Wentian model architecture."""

from math import ceil

import torch
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from torch import Tensor, nn
from torch.nn.init import trunc_normal_

from .layers import DropPath
from .utils import crop_2d, crop_3d, get_padded_shape_3d, pad_2d, pad_3d


def _normalize_lead_time(lead_time, reference: Tensor) -> Tensor:
    """Return one lead-time value per model batch element."""
    batch_size = reference.shape[0]
    if isinstance(lead_time, (int, float)):
        return reference.new_full((batch_size,), lead_time)
    if not torch.is_tensor(lead_time):
        raise TypeError("lead_time must be a number or torch.Tensor")
    lead_time = lead_time.to(device=reference.device, dtype=reference.dtype)
    if lead_time.ndim == 0:
        return lead_time.expand(batch_size)
    if lead_time.ndim != 1:
        raise ValueError("lead_time must be a scalar or one-dimensional tensor")
    if lead_time.shape[0] == 1:
        return lead_time.expand(batch_size)
    if lead_time.shape[0] != batch_size:
        raise ValueError(
            f"lead_time length must be 1 or batch size {batch_size}, got {lead_time.shape[0]}"
        )
    return lead_time


class WentianModel(nn.Module):
    def __init__(
        self,
        num_plevel_var: tuple = (5, 5),
        num_surface_var: tuple = (4, 4),
        num_pressure_level: int = 13,
        seq_len=1,
        depths: tuple = (2, 6, 6, 2),
        heads: tuple = (2, 6, 6, 2),
        dim: int = 192,
        image_size: tuple = (721, 1440),
        patch_size: tuple = (2, 4, 4),
        window_size: tuple = (2, 6, 12),
        drop_path_rate: float = 0.2,
        dropout_rate=0.0,
        use_checkpoint=True,
    ):
        super().__init__()
        block_num = len(depths)
        stage_num = block_num // 2
        if block_num != len(heads):
            raise ValueError("depths and heads must have the same length")
        if stage_num * 2 != block_num:
            raise ValueError("depths must define the same number of down and up stages")

        (input_num_plevel_var, output_num_plevel_var) = num_plevel_var
        (input_num_surface_var, output_num_surface_var) = num_surface_var

        image_size = (num_pressure_level,) + image_size
        down_depths, up_depths = depths[:stage_num], depths[stage_num:]
        down_heads, up_heads = heads[:stage_num], heads[stage_num:]

        down_drop_path_list = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(down_depths))
        ]
        up_drop_path_list = [x.item() for x in torch.linspace(drop_path_rate, 0, sum(up_depths))]

        self.has_surface = num_surface_var[0] > 0
        self.seq_len = seq_len
        if seq_len > 1:
            self.seq_len_linear = nn.Linear(seq_len * dim, dim)
        self.use_checkpoint = use_checkpoint

        # Patch embedding
        self.input_layer = PatchEmbedding(
            input_num_plevel_var, input_num_surface_var, dim, patch_size
        )

        def stage_drop_path_rates(stage_depths, index, drop_path_rates):
            start = sum(stage_depths[:index])
            return drop_path_rates[start : start + stage_depths[index]]

        down_blocks, up_blocks = [], []

        resolution = (
            ceil(image_size[0] / patch_size[0]) + int(self.has_surface),
            ceil(image_size[1] / patch_size[1]),
            ceil(image_size[2] / patch_size[2]),
        )

        for index in range(stage_num):
            # build DownBlock from top to bottom
            stage_dim = dim * (2**index)
            # downsample only effects on the height and width
            input_resolution = (
                resolution[0],
                ceil(resolution[1] / (2**index)),
                ceil(resolution[2] / (2**index)),
            )
            block_drop_path_rates = stage_drop_path_rates(down_depths, index, down_drop_path_list)
            down_blocks.append(
                TransformerLayer(
                    down_depths[index],
                    stage_dim,
                    block_drop_path_rates,
                    down_heads[index],
                    input_resolution,
                    window_size=window_size,
                    use_checkpoint=self.use_checkpoint,
                    dropout_rate=dropout_rate,
                )
            )
        self.down_blocks = nn.ModuleList(down_blocks)

        for index in range(stage_num):
            # build UpBlock from bottom to top
            stage_dim = dim * (2 ** (stage_num - index - 1))
            input_resolution = (
                resolution[0],
                ceil(resolution[1] / (2 ** (stage_num - index - 1))),
                ceil(resolution[2] / (2 ** (stage_num - index - 1))),
            )
            block_drop_path_rates = stage_drop_path_rates(up_depths, index, up_drop_path_list)
            up_blocks.append(
                TransformerLayer(
                    up_depths[index],
                    stage_dim,
                    block_drop_path_rates,
                    up_heads[index],
                    input_resolution,
                    window_size=window_size,
                    use_checkpoint=self.use_checkpoint,
                    dropout_rate=dropout_rate,
                )
            )
        self.up_blocks = nn.ModuleList(up_blocks)

        # Upsample and downsample
        upsamples, downsamples = [], []
        for index in range(stage_num - 1):
            # build UpSample from bottom to top
            stage_dim = dim * (2 ** (stage_num - index - 2))
            upsamples.append(UpSample(stage_dim * 2, stage_dim))
        self.upsamples = nn.ModuleList(upsamples)

        for index in range(stage_num - 1):
            # build DownSample from top to bottom
            stage_dim = dim * (2**index)
            downsamples.append(DownSample(stage_dim))
        self.downsamples = nn.ModuleList(downsamples)

        # Skip connection
        if stage_num > 2:
            skip_connections = []
            for index in range(stage_num - 2):
                stage_dim = dim * (2 ** (stage_num - index - 2))
                skip_connections.append(nn.Linear(stage_dim * 2, stage_dim))
            self.skip_connections = nn.ModuleList(skip_connections)
        else:
            self.skip_connections = None

        self.output_layer = PatchRecovery(
            output_num_plevel_var, output_num_surface_var, 2 * dim, image_size, patch_size
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, plevel, surface, lead_time):
        """Run the canonical Wentian forward pass."""
        reference = plevel if plevel is not None else surface
        if reference is None:
            raise ValueError("plevel or surface input must be provided")
        lead_time = _normalize_lead_time(lead_time, reference)

        if self.seq_len > 1:
            plevel = (
                rearrange(plevel, "b t c z h w -> (b t) c z h w") if plevel is not None else None
            )
            surface = (
                rearrange(surface, "b t  c h w -> (b t) c h w") if surface is not None else None
            )

        x = self.input_layer(plevel, surface)
        if self.seq_len > 1:
            x = rearrange(x, "(b t) z h w c -> b z h w (t c)", t=self.seq_len)
            x = self.seq_len_linear(x)

        encoder_features = []
        for index, block in enumerate(self.down_blocks):
            x = block(x, lead_time)
            encoder_features.append(x)
            if index < len(self.downsamples):
                x = self.downsamples[index](x)

        for index in range(len(self.up_blocks) - 1):
            if index >= 1:
                x = torch.concat([encoder_features[-index - 1], x], dim=-1)
                x = self.skip_connections[index - 1](x)
            x = self.up_blocks[index](x, lead_time)
            x = self.upsamples[index](x)
        x = self.up_blocks[-1](x, lead_time)
        x = crop_3d(x, encoder_features[0].shape[1:-1])
        x = torch.concat([encoder_features[0], x], dim=-1)
        output_plevel, output_surface = self.output_layer(x)
        return output_plevel, output_surface

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"abs_position_embedding"}

    def finetune_keywords(self):
        return [
            "input_layer.conv",
            "output_layer.conv",
            "input_layer.conv_surface",
            "output_layer.conv_surface",
        ]

    def interface_keywords(self):
        return self.finetune_keywords()


class TimeEmbedding(nn.Sequential):
    r"""Creates a time embedding.

    Arguments:
        features: The number of embedding features.
    """

    def __init__(self, features: int):
        super().__init__(
            nn.Linear(32, 256),
            nn.SiLU(),
            nn.Linear(256, features),
        )

        self.register_buffer("freqs", torch.pi * torch.arange(1, 16 + 1))

    def forward(self, t: Tensor) -> Tensor:
        t = self.freqs * t.unsqueeze(dim=-1)
        t = torch.cat((t.cos(), t.sin()), dim=-1)

        return super().forward(t)


class PatchEmbedding(nn.Module):
    def __init__(
        self,
        num_plevel_var: int = 5,
        num_surface_var: int = 4,
        dim: int = 192,
        patch_size: tuple = (2, 4, 4),
    ):
        """Patch embedding operation"""
        super().__init__()
        # Here we use convolution to partition data into cubes
        self.has_plevel = num_plevel_var > 0
        self.has_surface = num_surface_var > 0
        if self.has_plevel:
            self.conv = nn.Conv3d(
                in_channels=num_plevel_var,
                out_channels=dim,
                kernel_size=patch_size,
                stride=patch_size,
            )
        if self.has_surface:
            self.conv_surface = nn.Conv2d(
                in_channels=num_surface_var,
                out_channels=dim,
                kernel_size=patch_size[1:],
                stride=patch_size[1:],
            )

        # Set patch size attribute
        self.patch_size = patch_size

    def forward(self, plevel_x=None, surface_x=None):
        """Patch embedding operation
        input: Tensor shape (B C Z H W) Z for pressure level, H for latitude, W for longitude
        input_surface: Tensor shape (B C H W)
        """
        # Zero-pad the input
        if self.has_plevel:
            plevel_x = pad_3d(plevel_x, self.patch_size, channel_last=False)
            plevel_x = self.conv(plevel_x) if plevel_x is not None else None

        if self.has_surface:
            surface_x = pad_2d(surface_x, self.patch_size[1:], channel_last=False)
            surface_x = self.conv_surface(surface_x).unsqueeze(2)

        if plevel_x is not None and surface_x is not None:
            # Concatenate the input in the pressure level, i.e., in Z dimension
            x = torch.concat([plevel_x, surface_x], dim=2)
        elif plevel_x is not None:
            x = plevel_x
        else:
            x = surface_x
        x = rearrange(x, "b c z h w -> b z h w c")

        return x


class PatchRecovery(nn.Module):
    def __init__(
        self,
        num_plevel_var: int = 5,
        num_surface_var: int = 4,
        dim: int = 384,
        output_size: tuple = (14, 721, 1440),
        patch_size: tuple = (2, 4, 4),
    ):
        """Patch recovery operation"""
        super().__init__()
        # Hear we use two transposed convolutions to recover data
        self.has_plevel = num_plevel_var > 0
        self.has_surface = num_surface_var > 0
        if self.has_plevel:
            self.conv = nn.ConvTranspose3d(
                in_channels=dim,
                out_channels=num_plevel_var,
                kernel_size=patch_size,
                stride=patch_size,
            )
        if self.has_surface:
            self.conv_surface = nn.ConvTranspose2d(
                in_channels=dim,
                out_channels=num_surface_var,
                kernel_size=patch_size[1:],
                stride=patch_size[1:],
            )

        # Set the input size and patch size attribute
        self.output_size = output_size
        self.patch_size = patch_size

    def forward(self, x):
        x = rearrange(x, "b z h w c -> b c z h w")

        # Call the transposed convolution
        if self.has_plevel and self.has_surface:
            output = self.conv(x[:, :, :-1, :, :])
            output_surface = self.conv_surface(x[:, :, -1, :, :])
        elif self.has_plevel:
            output = self.conv(x)
            output_surface = None
        elif self.has_surface:
            output = None
            output_surface = self.conv_surface(x)
        else:
            output = None
            output_surface = None
        # Crop the output to remove zero-paddings
        output = crop_3d(output, self.output_size, channel_last=False) if self.has_plevel else None
        output_surface = (
            crop_2d(output_surface, self.output_size[1:], channel_last=False)
            if self.has_surface
            else None
        )
        return output, output_surface


class DownSample(nn.Module):
    def __init__(self, dim):
        """Down-sampling operation, similar with PatchMerging in Swin-Transformer"""
        super().__init__()
        # A linear function and a layer normalization
        self.linear = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x):
        x = pad_3d(x, (1, 2, 2))
        height, width = x.shape[2:4]

        x = rearrange(
            x,
            "b z (h k1) (w k2) c -> b z h w (k1 k2 c)",
            h=height // 2,
            w=width // 2,
            k1=2,
            k2=2,
        )

        # Call the layer normalization
        x = self.norm(x)

        # Decrease the channels of the data to reduce computation cost
        x = self.linear(x)

        return x


class UpSample(nn.Module):
    def __init__(self, input_dim, output_dim):
        """Up-sampling operation"""
        super().__init__()
        # Linear layers without bias to increase channels of the data
        self.linear1 = nn.Linear(input_dim, output_dim * 4, bias=False)

        # Linear layers without bias to mix the data up
        self.linear2 = nn.Linear(output_dim, output_dim, bias=False)

        # Normalization
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x):
        height, width, channels = x.shape[2:5]
        if channels % 4 != 0:
            raise ValueError("input channels must be divisible by 4")

        # Call the linear functions to increase channels of the data
        x = self.linear1(x)

        x = rearrange(
            x,
            "b z h w (k1 k2 c) -> b z (h k1) (w k2) c",
            h=height,
            w=width,
            c=channels // 2,
            k1=2,
            k2=2,
        )

        # Call the layer normalization
        x = self.norm(x)

        # Mixup normalized tensors
        x = self.linear2(x)
        return x


class TransformerLayer(nn.Module):
    def __init__(
        self,
        depth,
        dim,
        drop_path_ratio_list,
        heads,
        input_resolution,
        window_size,
        use_checkpoint=True,
        dropout_rate=0.0,
    ):
        """A stack of Wentian transformer blocks."""
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim,
                    drop_path_ratio_list[i],
                    heads,
                    input_resolution,
                    window_size,
                    roll=(i % 2 == 0),
                    dropout_rate=dropout_rate,
                )
                for i in range(depth)
            ]
        )

    def forward(self, x, lead_time):
        for blk in self.blocks:
            # Roll the input every two blocks
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, lead_time)
            else:
                x = blk(x, lead_time)
        return x


def window_partition(x, window_size):
    """Split a channel-last 3D tensor into non-overlapping windows."""
    z, h, w = window_size
    windows = rearrange(x, "b (zp z) (hp h) (wp w) c -> (b zp hp wp) z h w c", z=z, h=h, w=w)
    return windows


def window_reverse(windows, window_size, image_size):
    """Reassemble non-overlapping windows into a channel-last 3D tensor."""
    depth, height, width = image_size
    z, h, w = window_size
    x = rearrange(
        windows,
        "(b zp hp wp) z h w c -> b (zp z) (hp h) (wp w) c",
        zp=depth // z,
        hp=height // h,
        wp=width // w,
    )
    return x


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        drop_path_ratio,
        heads,
        input_resolution,
        window_size: tuple = (2, 6, 12),
        roll: bool = False,
        dropout_rate=0.0,
    ):
        """3D transformer block with Earth-specific window attention."""
        super().__init__()
        # Define the resolution of input
        self.input_resolution = input_resolution

        # Define the window size of the neural network
        self.window_size = window_size

        # Initialize serveral operations
        self.drop_path = DropPath(drop_prob=drop_path_ratio)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm_conv = nn.LayerNorm(dim)
        self.linear = Mlp(dim, dropout_rate)
        padded_input_resolution = get_padded_shape_3d(input_resolution, window_size)

        self.large_kernel_conv = nn.Conv2d(
            dim, dim, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=dim
        )

        self.attention = WindowAttention3D(
            dim,
            heads,
            dropout_rate,
            padded_input_resolution,
            self.window_size,
        )
        self.roll = roll
        if self.roll:
            self.shift_size = (window_size[0] // 2, window_size[1] // 2, window_size[2] // 2)
            depth, height, width = self.input_resolution
            img_mask = torch.zeros((1, depth, height, width, 1))
            img_mask = pad_3d(img_mask, window_size)
            cnt = 0
            # w do not need this attention mask, because this is a cyclic position
            for z in (
                slice(0, -self.window_size[0]),
                slice(-self.window_size[0], -self.shift_size[0]),
                slice(-self.shift_size[0], None),
            ):
                for h in (
                    slice(0, -self.window_size[1]),
                    slice(-self.window_size[1], -self.shift_size[1]),
                    slice(-self.shift_size[1], None),
                ):
                    img_mask[:, z, h, :, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = rearrange(mask_windows, "b z h w c -> b (z h w) c")
            mask_windows = mask_windows.squeeze(-1)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, (-100.0)).masked_fill(
                attn_mask == 0, 0.0
            )
        else:
            attn_mask = None

        self.lead_time_embedding = nn.Sequential(
            TimeEmbedding(dim), nn.Linear(dim, dim), nn.Unflatten(-1, (1, 1, 1, -1))
        )

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x, lead_time):
        # Save the shortcut for skip-connection
        shortcut = x

        # add lead time information
        t = self.lead_time_embedding(lead_time)
        # print("in transformer block, x shape:", x.shape, "t shape:", t.shape)
        x = x + t

        # Store the shape of the input for restoration
        ori_shape = x.shape

        x = pad_3d(x, self.window_size)
        batch, depth, height, width, _ = x.shape
        x = rearrange(x, "b z h w c -> (b z) c h w")
        x = x.contiguous()
        x = self.large_kernel_conv(x) + x
        x = rearrange(x, "(b z) c h w -> b z h w c", b=batch)
        x = x.contiguous()
        x = self.norm_conv(x)

        if self.roll:
            # Roll x for half of the window for 3 dimensions
            x = torch.roll(
                x,
                shifts=(-self.shift_size[0], -self.shift_size[1], -self.shift_size[2]),
                dims=(1, 2, 3),
            )
        x_window = rearrange(window_partition(x, self.window_size), "b z h w c -> b (z h w) c")

        # Apply 3D window attention with Earth-Specific bias

        attn_window = self.attention(x_window, self.attn_mask)

        attn_window = rearrange(
            attn_window,
            "b (z h w) c -> b z h w c",
            z=self.window_size[0],
            h=self.window_size[1],
            w=self.window_size[2],
        )

        x = window_reverse(attn_window, self.window_size, (depth, height, width))

        if self.roll:
            # Roll x back for half of the window
            x = torch.roll(
                x,
                shifts=(self.shift_size[0], self.shift_size[1], self.shift_size[2]),
                dims=(1, 2, 3),
            )

        # Crop the zero-padding
        x = crop_3d(x, ori_shape[1:-1])

        # Main calculation stages
        x = shortcut + self.drop_path(self.norm1(x))
        x = x + self.drop_path(self.norm2(self.linear(x)))
        return x


class WindowAttention3D(nn.Module):
    def __init__(self, dim, heads, dropout_rate, input_shape, window_size):
        super().__init__()
        # Initialize several operations
        self.linear1 = nn.Linear(dim, dim * 3, bias=True)
        self.linear2 = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout_rate)

        # Store several attributes
        self.head_number = heads
        self.scale = (dim // heads) ** -0.5
        self.window_size = window_size

        # input_shape is current shape of the self.forward function
        # You can run your code to record it, modify the code and rerun it
        # Record the number of different window types

        self.type_of_windows = (input_shape[0] // window_size[0]) * (
            input_shape[1] // window_size[1]
        )

        # For each type of window, we will construct a set of parameters according to the paper
        self.abs_position_embedding = torch.Tensor(
            size=(
                (2 * window_size[2] - 1)
                * window_size[1]
                * window_size[1]
                * window_size[0]
                * window_size[0],
                self.type_of_windows,
                heads,
            )
        )

        # Making these tensors to be learnable parameters
        self.abs_position_embedding = nn.Parameter(self.abs_position_embedding)

        # Initialize the tensors using Truncated normal distribution
        trunc_normal_(self.abs_position_embedding, std=0.02)

        # Construct position index to reuse self.abs_position_embedding
        self.register_buffer("position_index", self._construct_index(), persistent=False)

    def _construct_index(self):
        """Construct indices for reusing symmetric position-bias parameters."""
        # Index in the pressure level of query matrix
        coords_zi = torch.arange(self.window_size[0])
        # Index in the pressure level of key matrix
        coords_zj = -torch.arange(self.window_size[0]) * self.window_size[0]

        # Index in the latitude of query matrix
        coords_hi = torch.arange(self.window_size[1])
        # Index in the latitude of key matrix
        coords_hj = -torch.arange(self.window_size[1]) * self.window_size[1]

        # Index in the longitude of the key-value pair
        coords_w = torch.arange(self.window_size[2])

        # Change the order of the index to calculate the index in total
        coords_1 = torch.stack(torch.meshgrid(coords_zi, coords_hi, coords_w, indexing="ij"))
        coords_2 = torch.stack(torch.meshgrid(coords_zj, coords_hj, coords_w, indexing="ij"))
        coords_flatten_1 = torch.flatten(coords_1, start_dim=1)
        coords_flatten_2 = torch.flatten(coords_2, start_dim=1)
        coords = coords_flatten_1[:, :, None] - coords_flatten_2[:, None, :]
        coords = coords.permute(1, 2, 0).contiguous()

        # Shift the index for each dimension to start from 0
        coords[:, :, 2] += self.window_size[2] - 1
        coords[:, :, 1] *= 2 * self.window_size[2] - 1
        coords[:, :, 0] *= (2 * self.window_size[2] - 1) * self.window_size[1] * self.window_size[1]

        # Sum up the indexes in three dimensions
        position_index = torch.sum(coords, dim=-1)

        # Flatten the position index to facilitate further indexing
        return torch.flatten(position_index)

    def forward(self, x, mask):
        # Linear layer to create query, key and value
        x = self.linear1(x)

        # Record the original shape of the input
        original_shape = x.shape

        # reshape the data to calculate multi-head attention
        x = rearrange(x, "b s (k h d) -> k b h s d", k=3, h=self.head_number)
        query, key, value = torch.unbind(x, dim=0)

        # Scale the attention
        query = query * self.scale

        # Calculated the attention, a learnable bias is added to fix the nonuniformity of the grid.
        attention = query @ key.transpose(-2, -1)  # @ denotes matrix multiplication

        # self.abs_position_embedding is a set of neural network parameters to optimize.
        abs_position_embedding = self.abs_position_embedding[self.position_index]

        # Reshape the learnable bias to the same shape as the attention matrix
        seq_length = self.window_size[0] * self.window_size[1] * self.window_size[2]

        abs_position_embedding = rearrange(
            abs_position_embedding, "(n1 n2) t h -> t h n1 n2", n1=seq_length, n2=seq_length
        )

        # Add the Earth-Specific bias to the attention matrix
        b2 = abs_position_embedding.shape[0]
        attention = rearrange(attention, "(b1 b2) h s1 s2 -> b1 b2 h s1 s2", b2=b2)

        # b nd*nh*nw head s s + nd*nh*nw head s s -> b nd*nh*nw head s s
        attention = attention + abs_position_embedding

        if mask is not None:
            bs = original_shape[0] // mask.shape[0]
            attention = rearrange(attention, "b1 b2 h s1 s2 -> (b1 b2) h s1 s2")
            attention = rearrange(attention, "(b0 b1) h s1 s2 -> b0 b1 h s1 s2", b0=bs)
            attention = attention + mask.unsqueeze(0).unsqueeze(2)

        attention = rearrange(attention, "b1 b2 h s1 s2 -> (b1 b2) h s1 s2")

        attention = self.softmax(attention)
        attention = self.dropout(attention)

        # Calculated the tensor after spatial mixing.
        x = attention @ value  # @ denote matrix multiplication

        # Reshape tensor to the original shape
        x = rearrange(x, "b h s d -> b s (h d)")

        # Linear layer to post-process operated tensor
        x = self.linear2(x)
        x = self.dropout(x)
        return x


class Mlp(nn.Module):
    def __init__(self, dim, dropout_rate):
        """MLP layers, same as most vision transformer architectures."""
        super().__init__()
        self.linear1 = nn.Linear(dim, dim * 4)
        self.linear2 = nn.Linear(dim * 4, dim)
        self.activation = nn.GELU()
        self.drop = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.drop(x)
        x = self.linear2(x)
        x = self.drop(x)
        return x
