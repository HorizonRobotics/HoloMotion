"""Single-model validation, two-model matches and saved-trajectory playback."""

import argparse
import json
import os


def _resources(parser, *, lower=False, optional=False):
    assets = os.environ.get("HOLOBOXING_ASSETS")
    parser.add_argument(
        "--assets",
        default=assets,
        required=not optional and not bool(assets),
        help="explicit boxing resource directory; bundled in the participant image",
    )
    if lower:
        checkpoint = os.environ.get("HOLOBOXING_LOWER")
        parser.add_argument(
            "--lower",
            default=checkpoint,
            required=not optional and not bool(checkpoint),
            help="pinned public HoloMotion V2 bundle; bundled in the participant image",
        )


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    validate = commands.add_parser(
        "validate",
        help="check one ONNX model and run a CUDA compatibility probe",
    )
    validate.add_argument("model")
    _resources(validate, lower=True, optional=True)
    validate.add_argument("--output", default="validation.json")
    validate.add_argument(
        "--benchmark",
        action="store_true",
        help="also measure the complete upper + frozen-lower chain",
    )
    validate.add_argument(
        "--qualification",
        choices=("local", "official"),
        default="local",
        help="local checks compatibility; official always includes complete-chain timing on the required hardware",
    )
    match = commands.add_parser(
        "match", help="run both fighters on the first visible GPU"
    )
    match.add_argument(
        "red_model",
        metavar="YOUR_POLICY",
        help="Your model; controls the red robot",
    )
    match.add_argument(
        "blue_model",
        metavar="OPPONENT_POLICY",
        help="Opponent model; controls the blue robot",
    )
    _resources(match, lower=True)
    match.add_argument("--output-dir", default="results")
    view = commands.add_parser(
        "view", help="serve the existing replay player without running physics"
    )
    view.add_argument("replay", help="saved replay.hdf5 or match.json")
    _resources(view)
    view.add_argument("--host", default="127.0.0.1")
    view.add_argument("--port", type=int, default=8080)
    view.add_argument("--tls-cert", help="HTTPS certificate in PEM format")
    view.add_argument("--tls-key", help="HTTPS private key in PEM format")
    assets = commands.add_parser(
        "assets", help="verify resource bytes, hashes and relative references"
    )
    _resources(assets)
    download = commands.add_parser(
        "download", help="fetch the existing public V2 checkpoint bundle"
    )
    download.add_argument("--directory", required=True)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "assets":
            from .resources import ResourceTree

            print(json.dumps(ResourceTree(args.assets).verify(), indent=2))
        elif args.command == "download":
            from .models import download_bundle

            print(
                json.dumps(
                    {
                        "status": "PASS",
                        "directory": str(download_bundle(args.directory)),
                    }
                )
            )
        elif args.command == "validate":
            from .validation import validate_model

            report = validate_model(
                args.model,
                args.assets,
                args.lower,
                qualification_mode=args.qualification,
                full_chain=args.benchmark or args.qualification == "official",
            )
            from .io import write_json

            write_json(args.output, report)
            print(json.dumps(report, indent=2, allow_nan=False))
            return 0 if report["status"] == "PASS" else 1
        elif args.command == "match":
            from .match import evaluate_match

            report = evaluate_match(
                args.red_model,
                args.blue_model,
                args.assets,
                args.lower,
                args.output_dir,
            )
            print(json.dumps(report, indent=2, allow_nan=False))
        elif args.command == "view":
            from .viewer_server import serve

            serve(args.replay, args.assets, host=args.host, port=args.port,
                  tls_cert=args.tls_cert, tls_key=args.tls_key)
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as error:
        print(
            f"{type(error).__name__}: {error}", file=__import__("sys").stderr
        )
        return 1
