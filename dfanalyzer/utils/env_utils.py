import os


def get_bool_env_var(name: str, default: bool = False) -> bool:
    return os.environ.get(name, str(default)).lower() in ["true", "1", "yes"]


def get_int_env_var(name: str, default: int = 0) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default
