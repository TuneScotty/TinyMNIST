import tensorflow as tf
import os
from src.pipeline.utils.serialize import (
    to_lua_matrix, to_lua_vector,
    write_module, transpose, finite_check
)

MODEL_PATH = "src/pipeline/weight/mnist256.keras"
OUT_DIR = "src/pipeline/luau_weights"

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    model = tf.keras.models.load_model(MODEL_PATH)
    weighted_layers = [l for l in model.layers if len(l.get_weights()) > 0]

    W1, B1 = weighted_layers[0].get_weights()
    W2, B2 = weighted_layers[1].get_weights()

    W1 = transpose(W1.tolist())
    W2 = transpose(W2.tolist())
    B1 = B1.tolist()
    B2 = B2.tolist()

    finite_check([W1, B1, W2, B2])

    write_module(os.path.join(OUT_DIR, "W1.lua"), to_lua_matrix(W1))
    write_module(os.path.join(OUT_DIR, "B1.lua"), to_lua_vector(B1))
    write_module(os.path.join(OUT_DIR, "W2.lua"), to_lua_matrix(W2))
    write_module(os.path.join(OUT_DIR, "B2.lua"), to_lua_vector(B2))

    print("Export complete:")
    print(f"W1: {len(W1)}x{len(W1[0])}")
    print(f"B1: {len(B1)}")
    print(f"W2: {len(W2)}x{len(W2[0])}")
    print(f"B2: {len(B2)}")

if __name__ == "__main__":
    main()
