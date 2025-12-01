REQUIRED_KEYS = ["data", "model", "training"]


def validate_yaml_schema(cfg):
    for key in REQUIRED_KEYS:
        assert key in cfg, f"Missing key: {key}"
