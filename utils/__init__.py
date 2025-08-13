import os
import sys
import logging
from datetime import datetime
import atexit
import importlib
import importlib.util
from typing import Union, Type
import torchvision
from einops import rearrange
import torch
import torch.distributed as dist


class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

def init_logging(log_dir, rank):

    os.makedirs(log_dir, exist_ok=True)
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(log_dir, f"log_rank{rank}_{now}.txt")

    log_file = open(log_filename, "w", buffering=1)
    atexit.register(log_file.close)

    sys.stdout = Tee(sys.__stdout__, log_file)
    sys.stderr = Tee(sys.__stderr__, log_file)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(log_file)
        ]
    )

    print(f"[INFO] Logging initialized at: {log_filename}")
    return log_file


def import_custom_class(
    class_name: str,
    source: Union[str, os.PathLike], 
) -> Type:
    """
    Intelligently import CustomClass, supporting both installed packages and absolute paths
    
    Parameters:
        source: Either:
                1. Module import path (e.g., "package.module")
                2. Filesystem absolute path (e.g., "/path/to/module.py")
        class_name: Name of class to import (default: "CustomClass")
    
    Returns:
        Imported class object
    """
    # Case 1: Attempt import from installed package
    if not os.path.isabs(source) and not source.endswith('.py'):
        try:
            module = importlib.import_module(source)
            if hasattr(module, class_name):
                return getattr(module, class_name)
            else:
                raise ImportError(
                    f"Class '{class_name}' not found in module '{source}'. "
                    f"Available symbols: {[attr for attr in dir(module) if not attr.startswith('__')]}"
                )
        except ImportError:
            # If not a valid module path, try as file path
            pass
    
    # Case 2: Import from file path
    file_path = os.path.abspath(source)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Create unique module name based on file path
    module_name = os.path.splitext(os.path.basename(file_path))[0]
    unique_name = module_name # f"dynamic_{module_name}_{os.path.getmtime(file_path)}"
    
    # Check if already loaded
    if unique_name in sys.modules:
        module = sys.modules[unique_name]
    else:
        # Create new module
        spec = importlib.util.spec_from_file_location(unique_name, file_path)
        if spec is None:
            raise ImportError(f"Failed to create module spec from file: {file_path}")
        
        module = importlib.util.module_from_spec(spec)
        sys.modules[unique_name] = module
        
        try:
            spec.loader.exec_module(module)
        except Exception as e:
            # Clean cache on failure
            del sys.modules[unique_name]
            raise RuntimeError(f"Module execution failed: {e}") from e
    
    # Get target class
    if hasattr(module, class_name):
        return getattr(module, class_name)
    else:
        available = [attr for attr in dir(module) if not attr.startswith('__')]
        raise AttributeError(
            f"Class '{class_name}' not found in file '{file_path}'. "
            f"Available symbols: {', '.join(available)}"
        )



def save_video(tensor, save_path, fps=30):
    """
    Input tensor: shape c,t,h,w ranging from -1 to 1
    """
    tensor = ((tensor + 1) / 2 * 255).to(torch.uint8)
    tensor = rearrange(tensor, "c t h w -> t h w c")
    torchvision.io.write_video(save_path, tensor, fps=fps)

def zero_rank_print(s):
    if (not dist.is_initialized()) and (dist.is_initialized() and dist.get_rank() == 0): print("### " + s)
