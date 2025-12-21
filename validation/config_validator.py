def validate_model_config(cfg):
    required_keys = [
        "input_size", "input_channels",
        "conv_layers", "dense_layers", "learning_rate"
    ]

    for key in required_keys:
        if key not in cfg:
            raise ValueError(f"Missing key: {key}")

    if cfg["input_size"] <= 0:
        raise ValueError("input_size must be > 0")

    if len(cfg["conv_layers"]) == 0:
        raise ValueError("At least one conv layer required")

    for i, conv in enumerate(cfg["conv_layers"]):
        if conv["filters"] <= 0:
            raise ValueError(f"Invalid filters in conv layer {i}")
        if conv["kernel"] not in [3, 5]:
            raise ValueError(f"Invalid kernel in conv layer {i}")

    if cfg["learning_rate"] <= 0:
        raise ValueError("learning_rate must be > 0")

    return True
