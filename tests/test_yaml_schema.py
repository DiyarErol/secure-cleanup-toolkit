def test_yaml_schema():
    import yaml

    from tests.schema.default_schema import validate_yaml_schema
    with open("configs/default.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    validate_yaml_schema(cfg)
