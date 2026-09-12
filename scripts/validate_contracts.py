from pathlib import Path

from jsonschema.validators import validator_for


def main() -> None:
    for path in sorted(Path("schemas").glob("*.schema.json")):
        schema = __import__("json").loads(path.read_text())
        validator_for(schema).check_schema(schema)
    print("contracts valid")


if __name__ == "__main__":
    main()
