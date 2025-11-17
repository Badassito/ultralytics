# Bias Decay Implementation Guide

This document explains the modifications made to enable bias weight decay in Ultralytics YOLO.

## Summary of Changes

The implementation adds a new `bias_decay` parameter that allows applying L2 regularization (weight decay) to bias parameters during training. By default, this is set to `0.0` to maintain standard deep learning practices.

## Files Modified

### 1. `ultralytics/cfg/default.yaml`
**Line 98**: Added new configuration parameter
```yaml
bias_decay: 0.0 # (float) bias weight decay (L2 regularization for bias parameters)
```

### 2. `ultralytics/cfg/__init__.py`
**Line 171**: Registered the parameter in `CFG_FRACTION_KEYS`
```python
CFG_FRACTION_KEYS = frozenset(
    {  # fractional float arguments with 0.0<=values<=1.0
        "dropout",
        "lr0",
        "lrf",
        "momentum",
        "weight_decay",
        "bias_decay",  # NEW
        "warmup_momentum",
        "warmup_bias_lr",
        ...
    }
)
```

### 3. `ultralytics/engine/trainer.py`
Three key modifications:

**Line 922-923**: Added bias_decay extraction
```python
# Get bias decay from args (defaults to 0.0 if not set)
bias_decay = getattr(self.args, 'bias_decay', 0.0)
```

**Lines 951-955**: Updated optimizer initialization to use bias_decay
```python
if name in {"Adam", "Adamax", "AdamW", "NAdam", "RAdam"}:
    optimizer = getattr(optim, name, optim.Adam)(g[2], lr=lr, betas=(momentum, 0.999), weight_decay=bias_decay)
elif name == "RMSProp":
    optimizer = optim.RMSprop(g[2], lr=lr, momentum=momentum, weight_decay=bias_decay)
elif name == "SGD":
    optimizer = optim.SGD(g[2], lr=lr, momentum=momentum, nesterov=True, weight_decay=bias_decay)
```

**Line 966**: Updated logging message
```python
f"{len(g[1])} weight(decay=0.0), {len(g[0])} weight(decay={decay}), {len(g[2])} bias(decay={bias_decay})"
```

## How to Use

### Method 1: Python API
```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
model.train(
    data="coco8.yaml",
    epochs=100,
    weight_decay=0.0005,  # Weight decay for model weights
    bias_decay=0.0001,     # Weight decay for bias parameters
)
```

### Method 2: Command Line
```bash
yolo train model=yolo11n.pt data=coco8.yaml weight_decay=0.0005 bias_decay=0.0001
```

### Method 3: Configuration File
Create a custom config file `custom_config.yaml`:
```yaml
# Inherit all defaults
weight_decay: 0.0005
bias_decay: 0.0001
```

Then use it:
```bash
yolo train model=yolo11n.pt data=coco8.yaml cfg=custom_config.yaml
```

## Parameter Details

- **Parameter Name**: `bias_decay`
- **Type**: float (range: 0.0 to 1.0)
- **Default Value**: `0.0` (no bias regularization)
- **Purpose**: Apply L2 regularization to bias parameters
- **Typical Values**:
  - `0.0` (default) - No bias decay
  - `0.00001` to `0.0001` - Light bias regularization
  - Should typically be lower than `weight_decay`

## Verification

When training starts, check the optimizer initialization message:

**Before (with bias_decay=0.0)**:
```
optimizer: AdamW(lr=0.001, momentum=0.9) with parameter groups 81 weight(decay=0.0), 88 weight(decay=0.0005), 87 bias(decay=0.0)
```

**After (with bias_decay=0.0001)**:
```
optimizer: AdamW(lr=0.001, momentum=0.9) with parameter groups 81 weight(decay=0.0), 88 weight(decay=0.0005), 87 bias(decay=0.0001)
```

## Testing

Run the included test script:
```bash
python test_bias_decay.py
```

This will verify that the parameter is correctly applied during optimizer initialization.

## Technical Notes

### Parameter Groups
The optimizer uses three parameter groups:
- **g[0]**: Model weights (affected by `weight_decay`)
- **g[1]**: Batch normalization weights and logit_scale (no decay)
- **g[2]**: Bias parameters (affected by `bias_decay`)

### Backward Compatibility
The default value of `0.0` ensures that existing training scripts and configurations will continue to work exactly as before, with no bias regularization applied.

### Why Bias Decay is Usually Disabled
In standard deep learning practice, bias parameters are typically not regularized because:
1. They don't contribute to model capacity as much as weights
2. They serve to shift activation functions rather than scale features
3. Regularizing biases can sometimes harm performance

However, this implementation allows users to experiment with bias regularization if desired for their specific use case.

## Example Use Cases

### Light Regularization
```python
model.train(
    data="coco8.yaml",
    weight_decay=0.0005,
    bias_decay=0.00001,  # 10x smaller than weight_decay
)
```

### Moderate Regularization
```python
model.train(
    data="coco8.yaml",
    weight_decay=0.001,
    bias_decay=0.0001,  # 10x smaller than weight_decay
)
```

### Disable All Regularization
```python
model.train(
    data="coco8.yaml",
    weight_decay=0.0,
    bias_decay=0.0,
)
```

## References

The implementation follows PyTorch's optimizer API where weight_decay is applied per parameter group, allowing fine-grained control over L2 regularization.
