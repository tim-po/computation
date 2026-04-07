"""Wire protocol for distributed column-parallel inference.

Messages are sent over TCP as:
  [4B msg_type][4B payload_len][4B layer_idx][4B ndim][ndim*4B shape][payload]

Tensors are serialized as raw bytes. bf16 is reinterpreted as int16 bits
(numpy doesn't support bf16), then restored on receive.
"""

import asyncio
import struct
import torch
import numpy as np
from enum import IntEnum


class MsgType(IntEnum):
    REGISTER = 0        # worker -> coordinator: announce column index
    ACK = 1             # coordinator -> worker: confirm registration
    COL_STATE = 2       # coordinator -> worker: projected column state
    COMPRESSED_KV = 3   # worker -> coordinator: compressed K + V at merge
    AVERAGED_KV = 4     # coordinator -> worker: averaged K + V after merge
    FINAL_STATE = 5     # worker -> coordinator: final column state
    SHUTDOWN = 6        # coordinator -> worker: stop
    READY = 7           # coordinator -> workers: begin forward pass
    LOGITS = 8          # coordinator -> worker: final logits (optional)


# Dtype encoding for wire format
_DTYPE_TO_CODE = {
    torch.float32: 0,
    torch.float16: 1,
    torch.bfloat16: 2,
    torch.int8: 3,
    torch.int16: 4,
    torch.int32: 5,
    torch.int64: 6,
}
_CODE_TO_DTYPE = {v: k for k, v in _DTYPE_TO_CODE.items()}

# numpy dtype mapping (bf16 goes through int16 reinterpret)
_DTYPE_TO_NP = {
    torch.float32: np.float32,
    torch.float16: np.float16,
    torch.int8: np.int8,
    torch.int16: np.int16,
    torch.int32: np.int32,
    torch.int64: np.int64,
}


def _tensor_to_bytes(t: torch.Tensor) -> tuple[bytes, int]:
    """Serialize tensor to raw bytes. Returns (bytes, dtype_code).
    bf16 tensors are reinterpreted as int16 bits for serialization."""
    original_dtype = t.dtype
    if original_dtype == torch.bfloat16:
        # Reinterpret bf16 bits as int16 (same size, numpy-compatible)
        t_cpu = t.detach().cpu().view(torch.int16)
        dtype_code = _DTYPE_TO_CODE[torch.bfloat16]
    else:
        t_cpu = t.detach().cpu()
        dtype_code = _DTYPE_TO_CODE[original_dtype]

    np_dtype = _DTYPE_TO_NP.get(t_cpu.dtype, np.float32)
    data = t_cpu.numpy().astype(np_dtype).tobytes()
    return data, dtype_code


def _bytes_to_tensor(data: bytes, shape: tuple, dtype_code: int, device: str = "cpu") -> torch.Tensor:
    """Deserialize raw bytes back to tensor."""
    dtype = _CODE_TO_DTYPE[dtype_code]

    if dtype == torch.bfloat16:
        # Was sent as int16 bits, restore bf16
        arr = np.frombuffer(data, dtype=np.int16).reshape(shape)
        t = torch.from_numpy(arr.copy()).view(torch.bfloat16)
    else:
        np_dtype = _DTYPE_TO_NP[dtype]
        arr = np.frombuffer(data, dtype=np_dtype).reshape(shape)
        t = torch.from_numpy(arr.copy())

    return t.to(device)


async def send_msg(
    writer: asyncio.StreamWriter,
    msg_type: MsgType,
    tensor: torch.Tensor | None = None,
    layer_idx: int = 0,
) -> None:
    """Send a message with optional tensor payload."""
    if tensor is not None:
        data, dtype_code = _tensor_to_bytes(tensor)
        shape = tensor.shape
        ndim = len(shape)
        # Header: msg_type(4) + payload_len(4) + layer_idx(4) + dtype_code(4) + ndim(4) + shape(ndim*4)
        header = struct.pack(
            f"!5I{ndim}I",
            int(msg_type),
            len(data),
            layer_idx,
            dtype_code,
            ndim,
            *shape,
        )
    else:
        # No payload
        header = struct.pack("!5I", int(msg_type), 0, layer_idx, 0, 0)
        data = b""

    writer.write(header + data)
    await writer.drain()


async def recv_msg(
    reader: asyncio.StreamReader,
    device: str = "cpu",
) -> tuple[MsgType, torch.Tensor | None, int]:
    """Receive a message. Returns (msg_type, tensor_or_None, layer_idx)."""
    # Read fixed header: msg_type + payload_len + layer_idx + dtype_code + ndim
    fixed_header = await reader.readexactly(20)  # 5 * 4 bytes
    msg_type_int, payload_len, layer_idx, dtype_code, ndim = struct.unpack("!5I", fixed_header)
    msg_type = MsgType(msg_type_int)

    if payload_len == 0:
        return msg_type, None, layer_idx

    # Read shape
    shape_data = await reader.readexactly(ndim * 4)
    shape = struct.unpack(f"!{ndim}I", shape_data)

    # Read payload
    data = await reader.readexactly(payload_len)
    tensor = _bytes_to_tensor(data, shape, dtype_code, device)

    return msg_type, tensor, layer_idx


async def send_tensor_pair(
    writer: asyncio.StreamWriter,
    msg_type: MsgType,
    t1: torch.Tensor,
    t2: torch.Tensor,
    layer_idx: int = 0,
) -> None:
    """Send two tensors (e.g., K and V) as a single concatenated message.
    Both tensors must have the same shape. They are concatenated along dim 0."""
    combined = torch.cat([t1, t2], dim=0)
    await send_msg(writer, msg_type, combined, layer_idx)


async def recv_tensor_pair(
    reader: asyncio.StreamReader,
    device: str = "cpu",
) -> tuple[MsgType, torch.Tensor, torch.Tensor, int]:
    """Receive two tensors that were sent with send_tensor_pair.
    Splits the concatenated tensor back into two equal halves along dim 0."""
    msg_type, combined, layer_idx = await recv_msg(reader, device)
    assert combined is not None, f"Expected tensor pair, got empty payload for {msg_type}"
    mid = combined.shape[0] // 2
    t1 = combined[:mid]
    t2 = combined[mid:]
    return msg_type, t1, t2, layer_idx
