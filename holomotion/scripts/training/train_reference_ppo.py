"""Train a simulator PPO control on the shared G1 reference-tracking task."""

import argparse
from dataclasses import MISSING, fields
from pathlib import Path

from holomotion.src.algo.reference_ppo.config import ReferencePPOConfig
from holomotion.src.algo.reference_ppo.training import execute


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in fields(ReferencePPOConfig):
        name = "--" + field.name.replace("_", "-")
        if field.default is MISSING:
            parser.add_argument(name, type=Path, required=True)
        elif type(field.default) is bool:
            parser.add_argument(
                name,
                action=argparse.BooleanOptionalAction,
                default=field.default,
            )
        else:
            parser.add_argument(
                name, type=type(field.default), default=field.default
            )
    try:
        return ReferencePPOConfig.from_mapping(vars(parser.parse_args()))
    except (ValueError, TypeError) as error:
        parser.error(str(error))


def main():
    execute(arguments())


if __name__ == "__main__":
    main()
