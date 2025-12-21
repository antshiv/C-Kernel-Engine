import ctypes
import os


def _root_dir() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(here)


def find_lib(*names: str) -> str:
    root = _root_dir()
    for name in names:
        build_path = os.path.join(root, "build", name)
        if os.path.exists(build_path):
            return build_path
        root_path = os.path.join(root, name)
        if os.path.exists(root_path):
            return root_path
    raise FileNotFoundError(f"Could not find any of: {', '.join(names)}")


def load_lib(*names: str):
    return ctypes.cdll.LoadLibrary(find_lib(*names))
