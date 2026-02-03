import re
import json
import torch
import torch.nn as nn
import torch.distributed as dist
from pathlib import Path
from safetensors import safe_open
from safetensors.torch import save_file
from typing import Optional, Dict, Union, List
try:
    from runai_model_streamer import SafetensorsStreamer
except ImportError:
    SafetensorsStreamer = None
from transformers import AutoConfig
from typing import Any

def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b

import contextlib
import torch.nn.init as init
from accelerate import init_empty_weights

def _dist_rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def _dist_world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


class Rank0First(contextlib.AbstractContextManager):
    def __enter__(self):
        if _dist_world_size() > 1 and _dist_rank() != 0:
            dist.barrier()
    
    def __exit__(self, exc_type, exc_value, traceback):
        if _dist_world_size() > 1 and _dist_rank() == 0:
            dist.barrier()

def hf_name_to_model_name(name: str, key_mapping: Optional[Dict[str, str]] = None) -> str:
    if key_mapping is None:
        return name
    for pattern, replacement in key_mapping.items():
        param_name, n_replace = re.subn(pattern, replacement, name)
        if n_replace > 0:
            return param_name
    return name

def set_meta_param(model: nn.Module, name: str, param: torch.Tensor, requires_grad: bool = True):
    parts = name.split(".")
    mod = model
    for p in parts[:-1]:
        mod = getattr(mod, p)
    setattr(mod, parts[-1], torch.nn.Parameter(param, requires_grad=requires_grad))
    return getattr(mod, parts[-1])

class B10ModelLoader:
    def __init__(self, b10fs_path: Path, num_shards: int, padding_size: int = 256):
        self._b10fs_path = Path(b10fs_path)
        self._b10fs_path.mkdir(parents=True, exist_ok=True)
        self.num_shards = num_shards
        self.padding_size = padding_size
    
    def empty_init_model(self, Module, model_path: Path):
        with init_empty_weights():
            config = AutoConfig.from_pretrained(model_path)
            model = Module(config)
            return model
    
    def b10fs_path(self, path: Union[Path, str]):
        return self._b10fs_path / Path(path)

    @torch.no_grad()
    def create_or_read_metadata_for_load(self, model: nn.Module, meta_path: Path, model_path: Optional[Path] = None, key_mapping: Optional[Dict[str, str]] = None):
        if meta_path.exists():
            with open(meta_path, "rb") as f:
                return json.load(f)
        param_numels = [p.numel() for _, p in model.named_parameters(remove_duplicate=False)]
        num_params = sum(param_numels)
        max_shard_size = ceil_div(num_params, self.padding_size) * self.padding_size // self.num_shards
        metadata: Dict[str, Any] = {
            "num_shards": self.num_shards,
            "max_shard_size": max_shard_size,
            "num_params": num_params,
        }
        param_info = {} # param_name: str -> shard_id: int
        cur_shard_id, cur_shard_size = 0, 0
        # Try to assign each param to a shard, and if the shard is full, increment the shard id and continue
        # This method can make sure each shard is almost balanced by controling the max shard size
        model_index, shard_id2model_index = None, None
        state_dict = sorted(list(model.named_parameters(remove_duplicate=False)), key=lambda x: x[1].numel(), reverse=True)
        if model_path is not None:
            index_path_list = list(model_path.rglob("*.safetensors.index.json"))
            if len(index_path_list) == 0:
                print(f"No *.safetensors.index.json file found in {model_path}")
            elif len(index_path_list) > 1:
                print(f"Multiple *.safetensors.index.json files found in {model_path}: {index_path_list}, skipping")
            else:
                index_path = index_path_list[0]
                with open(index_path, "r") as f:
                    hf_model_index = json.load(f)["weight_map"]
                model_index = dict((hf_name_to_model_name(name, key_mapping), idx) for name, idx in hf_model_index.items())
                state_dict = sorted(state_dict, key=lambda x: model_index.get(x[0], ""))
                shard_id2model_index = [set() for _ in range(self.num_shards)]
        for name, param in state_dict:
            param_info[name] = cur_shard_id
            if model_index is not None and shard_id2model_index is not None:
                if name not in model_index:
                    print(f"Param {name} not found in model_index, skipping")
                    continue
                shard_id2model_index[cur_shard_id].add(model_index.get(name, ""))
            cur_shard_size += param.numel()
            if cur_shard_size  >= max_shard_size:
                cur_shard_id += 1
                cur_shard_size = 0
        metadata["param_info"] = param_info
        if shard_id2model_index is not None:
            shard_id2model_index = [list(shard_id2model_index[i]) for i in range(self.num_shards)]
        metadata["shard_id2model_index"] = shard_id2model_index
        with open(meta_path, "w") as f:
            json.dump(metadata, f)
        return metadata
    
    def _load_safetensors_by_streamer(
        self, 
        shard_safetensors_path: List[str], 
        param_info: dict, 
        shard_id: int, 
        device: str = "cuda", 
        dtype: torch.dtype = torch.float32, 
        allow_missing: bool = False, 
        key_mapping: Optional[Dict[str, str]] = None
    ):
        if SafetensorsStreamer is None:
            raise ImportError("runai_model_streamer is not installed; cannot use SafetensorsStreamer.")
        name2loaded_param = {}
        with SafetensorsStreamer() as streamer:
            streamer.stream_files(shard_safetensors_path)
            for name, tensor in streamer.get_tensors():
                param_name = hf_name_to_model_name(name, key_mapping)
                if param_name not in param_info:
                    print(f"Param {name} not found in metadata, skipping")
                    if not allow_missing:
                        raise ValueError(f"Param {name} not found in metadata, skipping")
                    continue
                if param_info[param_name] == shard_id:
                    if tensor.dtype != torch.uint8:
                        name2loaded_param[param_name] = tensor.to(device, dtype=dtype, non_blocking=True)
                    else:
                        name2loaded_param[param_name] = tensor.to(device, non_blocking=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return name2loaded_param
    
    def _load_safetensors_by_file(
        self, 
        shard_safetensors_path: List[str], 
        param_info: dict, 
        shard_id: int, 
        device: str = "cuda", 
        dtype: torch.dtype = torch.float32, 
        allow_missing: bool = False, 
        key_mapping: Optional[Dict[str, str]] = None
    ):
        name2loaded_param = {}
        for p in shard_safetensors_path:
            with safe_open(str(p), framework="pt", device="cpu") as f:
                file_keys = set(f.keys())
                for name in file_keys:
                    param_name = hf_name_to_model_name(name, key_mapping)
                    if param_name not in param_info:
                        print(f"Param {name} not found in metadata, skipping")
                        if not allow_missing:
                            raise ValueError(f"Param {name} not found in metadata, skipping")
                        continue
                    tensor = f.get_tensor(name)
                    if param_info[param_name] == shard_id:
                        if tensor.dtype != torch.uint8:
                            name2loaded_param[param_name] = tensor.to(device, dtype=dtype, non_blocking=True)
                        else:
                            name2loaded_param[param_name] = tensor.to(device, non_blocking=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return name2loaded_param

    @torch.no_grad()
    def load_model_from_safetensors(
        self,
        model: torch.nn.Module, 
        shard_safetensors_dir: Path, 
        metadata: dict, 
        shard_id: int, 
        device: str = "cuda", 
        dtype: torch.dtype = torch.float32, 
        allow_missing: bool = True, 
        key_mapping: Optional[Dict[str, str]] = None,
        use_safetensors_streamer: bool = True,
    ):
        print(f"[{_dist_rank()}][B10ModelLoader]load_model_from_safetensors for {model.__class__.__name__}", flush=True)
        """
        Saves parameters AND buffers to a single .safetensors file on CPU.
        Keys match model.state_dict() exactly.
        """
        if shard_safetensors_dir.is_dir():
            shard_safetensors_path = [str(p) for p in shard_safetensors_dir.glob("*.safetensors")]
        else:
            assert str(shard_safetensors_dir).endswith(".safetensors"), f"{shard_safetensors_dir} should be a directory or a .safetensors file"
            shard_safetensors_path = [str(shard_safetensors_dir)]
        if metadata.get("shard_id2model_index", None) is not None:
            print(f"[B10ModelLoader]{shard_id=} load {metadata['shard_id2model_index'][shard_id]} only")
            shard_safetensors_path = list(filter(lambda p: Path(p).name in metadata["shard_id2model_index"][shard_id], shard_safetensors_path))
        if len(shard_safetensors_path) == 0:
            raise FileNotFoundError(f"No .safetensors file found in {shard_safetensors_dir}")
        if use_safetensors_streamer and SafetensorsStreamer is not None:
            name2loaded_param = self._load_safetensors_by_streamer(shard_safetensors_path, metadata["param_info"], shard_id, device, dtype, allow_missing, key_mapping)
        else:
            name2loaded_param = self._load_safetensors_by_file(shard_safetensors_path, metadata["param_info"], shard_id, device, dtype, allow_missing, key_mapping)
        state_dict = dict(model.named_parameters(remove_duplicate=False))
        sorted_names = sorted(state_dict.keys())
        handles = []
        for name in sorted_names:
            param = state_dict[name]
            param_dtype = dtype if param.data.dtype != torch.uint8 else param.data.dtype
            new_param = name2loaded_param[name] if name in name2loaded_param else torch.empty_like(param, device=device, dtype=param_dtype)
            if param.device.type == "meta":
                param = set_meta_param(model, name, new_param, param.requires_grad)
            else:
                param.data = new_param
            # print(f"[{dist.get_rank()}][B10ModelLoader]{name} {param.data.dtype}, {param.data.shape}", flush=True)
            if _dist_world_size() > 1:
                handle = dist.broadcast(param.data, src=metadata["param_info"][name], async_op=True)
                handles.append(handle)
        for handle in handles:
            handle.wait()

    @torch.no_grad()
    def save_model_shard(self,  model: torch.nn.Module, model_path: Path, metadata: dict, shard_id: int):
        """
        Saves parameters to a single .safetensors file on CPU.
        """
        if model_path.exists():
            return
        model_path.parent.mkdir(parents=True, exist_ok=True)
        state_dict = {}
        for name, param in model.named_parameters(remove_duplicate=False):
            if not isinstance(param, torch.Tensor):
                raise TypeError(f"Unexpected non-tensor in state_dict at key '{name}': {type(param)}")
            if metadata["param_info"][name] == shard_id:
                state_dict[name] = param.detach().to("cpu", copy=True).contiguous()
        save_file(state_dict, str(model_path))
