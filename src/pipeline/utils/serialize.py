import math
import os
from typing import Any

def to_lua_number(x: float) -> str:
    if not math.isfinite(x):
        raise ValueError(f"Non-finite number: {x}")
    s = f"{float(x):.6f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s

def to_lua_vector(v: list[float], wrap: int | None = 100) -> str:
    if not v:
        return "{}"
    nums = [to_lua_number(x) for x in v]
    if not wrap:
        return "{" + ",".join(nums) + "}"
    lines = []
    for i in range(0, len(nums), wrap):
        lines.append(",".join(nums[i:i+wrap]))
    return "{" + ",\n".join(lines) + "}"


def to_lua_matrix(m: list[list[float]], wrap: int | None = 8) -> str:
    rows = [to_lua_vector(row, wrap=wrap) for row in m]
    return "{\n" + ",\n".join(rows) + "\n}"

def write_module(path: str, lua_body: str) -> None:
    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("return " + lua_body + "\n")

def transpose(m: list[list[float]]) -> list[list[float]]:
    return [list(row) for row in zip(*m)]

def finite_check(obj: Any) -> None:
    if isinstance(obj, (int, float)):
        if not math.isfinite(obj):
            raise ValueError(f"Non-finite value detected: {obj}")
    elif isinstance(obj, list):
        for v in obj:
            finite_check(v)
    else:
        raise TypeError(f"Unsupported type in finite_check: {type(obj)}")