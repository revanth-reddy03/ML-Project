"""Open and inspect a joblib/pickle model file from this project."""
from pathlib import Path
import argparse
import sys

import joblib


def describe_object(path: Path) -> None:
    """Load a trusted serialized object and print a useful summary."""
    try:
        obj = joblib.load(path)
    except Exception as error:
        print(f"Could not open {path}: {error}", file=sys.stderr)
        sys.exit(1)

    print(f"File: {path}")
    print(f"Type: {type(obj).__module__}.{type(obj).__name__}")

    for attribute in ("feature_names_in_", "n_features_in_", "n_estimators", "classes_"):
        if hasattr(obj, attribute):
            value = getattr(obj, attribute)
            if hasattr(value, "tolist"):
                value = value.tolist()
            print(f"{attribute}: {value}")

    if hasattr(obj, "get_params"):
        print("Parameters:")
        for name, value in obj.get_params(deep=False).items():
            print(f"  {name}={value}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Open and inspect a .pkl model file.")
    parser.add_argument("file", type=Path, help="Path to the .pkl file")
    args = parser.parse_args()

    if args.file.suffix.lower() != ".pkl":
        parser.error("the file must have a .pkl extension")
    if not args.file.is_file():
        parser.error(f"file not found: {args.file}")

    describe_object(args.file)


if __name__ == "__main__":
    main()