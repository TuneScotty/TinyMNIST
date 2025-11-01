
<p align="center">
  <img src="./TinyMNIST.png" alt="TinyMNIST" width="350"/>
</p>

# TinyMNIST

A minimalist MNIST digit recognizer running entirely in Roblox. Features a clean drawing interface with real-time neural network inference implemented in pure Luau with no external ML libraries on the client.

![Demo](https://img.shields.io/badge/status-working-brightgreen) ![Python](https://img.shields.io/badge/python-3.8+-blue) ![License](https://img.shields.io/badge/license-MIT-green)

**[Play on Roblox](https://www.roblox.com/games/75201611287780/MNIST-256-Neurons)**

## What is this?

End-to-end machine learning in Roblox: train a neural network in Python, serialize the weights to Luau, and run inference entirely client-side. Draw digits 0-9 on a 28×28 canvas and watch the model predict in real-time.

**Tech stack:**
- **Training**: TensorFlow/Keras (Python)
- **Inference**: Pure Luau (no external libraries)
- **UI**: Minimalist design with real-time prediction grid
- **Deployment**: Automated .rbxl builds via GitHub Actions

## Architecture

### Model
- **Input**: 784 features (28×28 grayscale pixels)
- **Hidden**: 256 neurons, ReLU activation
- **Output**: 10 neurons, softmax (digits 0-9)
- **Parameters**: ~206k weights
- **Accuracy**: 98.2% on MNIST test set

### Pipeline
1. `train_mnist.py` → Train 2-layer feedforward network
2. `export_luau.py` → Serialize weights to `.lua` modules
3. `Model.luau` → Load weights and run forward pass
4. `init.client.luau` → Canvas, preprocessing, display

## The Process

### 64 → 256 Neurons: Why Scale Up?

**Initial attempt (64 neurons):**
- Test accuracy: 97.8%
- Training time: ~30s (10 epochs)
- Problem: Struggled with digits 4, 7, and especially 9

Analyzed per-class accuracy and found the 64-neuron model lacked capacity for fine-grained feature distinction. Similar-looking digits (9 vs 4, 7 vs 1) had high confusion rates.

**Scaling to 256 neurons:**
- Test accuracy: 98.2% (+0.4%)
- Training time: ~35s (11 epochs with early stopping)
- Result: Much better edge-case handling

The 4× parameter increase (16k → 66k in hidden layer) improved generalization without significant training overhead. Early stopping kicked in at epoch 11 when validation loss plateaued the model learned efficiently.

### Training Insights

**Early stopping behavior:**
- Best validation loss at epoch 8: 0.0595
- Patience set to 3 epochs
- Stopped at epoch 11 after no improvement
- Final model restored to epoch 8 weights

This suggests the 256-neuron architecture finds good solutions quickly. No need for 30+ epochs the model saturates around epoch 10.

**Data augmentation experiments:**
Initially tested rotation (±10°) and translation (±10%) to improve robustness. However, MNIST digits are fairly standardized and the model achieved 98%+ accuracy without augmentation. Opted for simplicity in the final version.

**Regularization:**
Experimented with 0.2 dropout between layers but found it unnecessary for this dataset. The small architecture and early stopping provided sufficient regularization.

### Weight Serialization

Weights are transposed during export because TensorFlow stores matrices column-major while Luau expects row-major for efficient dot products:

```python
W1 = transpose(W1.tolist())  # (784, 256) → (256, 784)
W2 = transpose(W2.tolist())  # (256, 10) → (10, 256)
```

This avoids runtime transposition overhead. Serialized weights are formatted with line wrapping for readability (~100 values per line).

### Luau Inference Engine

Pure Luau implementation no FFI, no external libs:
- Matrix-vector multiplication: Nested loops, row-wise
- ReLU: `v[i] = v[i] < 0 and 0 or v[i]`
- Softmax: Max subtraction for numerical stability

**Performance:** 5-10ms per inference in Roblox Studio (client-side).

### Canvas Preprocessing

The drawing canvas stores normalized pixel intensities (0-1 range):
- Brush size: 2 pixels with circular falloff
- Values clamped to [0, 1] to match training data
- No downsampling needed (already 28×28)

**Critical bug discovered:** Early implementation had double normalization storing pixels as 0-1 then dividing by 255 again before inference. This resulted in max input values of ~0.004 instead of ~1.0, causing the model to predict the same digit constantly (usually 5, the most common MNIST class). Fixed by storing normalized values directly in `canvasData`.

## Project Structure

```ru
TinyMNIST/
├── src/
│   ├── client/init.client.luau       # UI and canvas
│   ├── Core/
│   │   ├── Model.luau                 # NN inference
│   │   ├── MathOps.luau               # Matrix ops
│   │   └── Activations.luau           # ReLU, softmax
│   └── pipeline/
│       ├── train_mnist.py             # Training script
│       ├── export_luau.py             # Weight export
│       ├── luau_weights/              # Serialized weights
│       └── utils/serialize.py         # Python→Luau converter
├── default.project.json
└── README.md
```

## Challenges Solved

### Model always predicts digit 5
**Cause:** Weight checkpoint mismatch or incorrect normalization.  
**Fix:** Export from `best_mnist.keras` (ModelCheckpoint with best val_loss) and verify `canvasData` stores 0-1 values.

### Predictions stuck at ~10% for all digits
**Cause:** Double normalization pixels divided by 255 twice.  
**Fix:** Remove `normalizePixel()` in `getCanvasArray()` since data is already 0-1.

### Drawing invisible on canvas
**Cause:** Intensity clamping applied incorrectly, display values near-zero.  
**Fix:** Store normalized values (0-1) in `canvasData`, multiply by 255 only for display.

### Digits 4, 7, 9 frequently misclassified
**Cause:** 64-neuron model insufficient capacity.  
**Fix:** Scale to 256 neurons (+0.4% accuracy).

## Performance Metrics

- **Training**: 35s for 11 epochs (Intel i5, CPU only)
- **Export**: <5s for weight serialization
- **Inference**: 5-10ms per prediction (Roblox client)
- **Model size**: 1.6MB serialized Luau files

## Technical Notes

**Why client-side inference?** Demonstrates that neural networks can run efficiently in unconventional environments. For production ML in Roblox, server-side HTTP APIs are recommended for better model security and scalability.

**Why MNIST?** Small dataset (60k training images), fast training, well-understood baseline. Perfect for proof-of-concept.

## License

MIT License - see LICENSE file.

## Contributing

Contributions welcome. Areas of interest:
- CNN layers for better feature extraction
- Optimized matrix operations (vectorization)
- Touch controls for mobile
- Confidence thresholding and uncertainty display

Open an issue or submit a PR.

---

Built to explore ML deployment in non-traditional environments. For questions or feedback, open an issue.