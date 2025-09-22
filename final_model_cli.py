import argparse
import os
import sys
from typing import List, Optional

import pandas as pd


def _load_model(model_path: str):
    """Load a sklearn Pipeline saved via joblib or pickle."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    try:
        import joblib  # type: ignore
        return joblib.load(model_path)
    except Exception:
        import pickle
        with open(model_path, 'rb') as f:
            return pickle.load(f)


def _get_expected_features(pipeline) -> Optional[List[str]]:
    """Try to infer base feature names the pipeline expects, for validation and ordering."""
    # sklearn >=1.0: Pipeline exposes feature_names_in_
    if hasattr(pipeline, 'feature_names_in_'):
        return list(pipeline.feature_names_in_)
    # Try imputer step
    try:
        imputer = pipeline.named_steps.get('imputer')
        if hasattr(imputer, 'feature_names_in_'):
            return list(imputer.feature_names_in_)
    except Exception:
        pass
    return None


def cmd_info(args: argparse.Namespace) -> int:
    """Print model information and brief summary."""
    model_path = args.model
    try:
        pipeline = _load_model(model_path)
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        return 2

    expected_features = _get_expected_features(pipeline)
    print(f"Model: {model_path}")
    if expected_features is not None:
        print(f"Expected base features: {len(expected_features)}")
    else:
        print("Expected base features: <unknown>")
    return 0


def cmd_features(args: argparse.Namespace) -> int:
    """List expected input feature names (columns) for prediction."""
    try:
        pipeline = _load_model(args.model)
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        return 2
    features = _get_expected_features(pipeline)
    if not features:
        print("Could not determine expected features for the model.", file=sys.stderr)
        return 3
    if args.output is not None and args.output != '':
        out_path = args.output
        try:
            pd.Series(features, name='feature').to_csv(
                out_path, index=False, encoding='utf-8-sig')
            print(f"Saved feature list to: {out_path}")
        except Exception as e:
            print(f"Error writing feature list: {e}", file=sys.stderr)
            return 4
    else:
        for name in features:
            print(name)
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    """Batch predict probabilities (and labels) for an input CSV using the final model.

    The input CSV must contain all expected base features with matching names.
    """
    try:
        pipeline = _load_model(args.model)
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        return 2

    expected_features = _get_expected_features(pipeline)
    if not expected_features:
        print("Model does not expose expected feature names; cannot validate input.", file=sys.stderr)
        return 3

    input_path = args.input
    if not os.path.exists(input_path):
        print(f"Input file not found: {input_path}", file=sys.stderr)
        return 4

    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        print(f"Failed to read input CSV: {e}", file=sys.stderr)
        return 5

    missing = [c for c in expected_features if c not in df.columns]
    if missing:
        print(
            "Input CSV missing required columns: " + ", ".join(missing),
            file=sys.stderr,
        )
        return 6

    X = df[expected_features]
    # Predict probabilities and labels
    try:
        proba = pipeline.predict_proba(X)[:, 1]
    except Exception as e:
        print(f"Prediction failed: {e}", file=sys.stderr)
        return 7

    threshold = args.threshold
    preds = (proba >= threshold).astype(int)

    result = pd.DataFrame({'prob_positive': proba, 'pred_label': preds})
    # Keep ID column if provided and present
    if args.id_column and args.id_column in df.columns:
        result.insert(0, args.id_column, df[args.id_column].values)

    out_path = args.output
    if not out_path:
        base, ext = os.path.splitext(os.path.basename(input_path))
        out_path = os.path.join(os.path.dirname(
            input_path), f"{base}_pred.csv")

    try:
        result.to_csv(out_path, index=False, encoding='utf-8-sig')
    except Exception as e:
        print(f"Writing predictions failed: {e}", file=sys.stderr)
        return 8

    print(f"Saved predictions to: {out_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CLI wrapper for the final Logistic Regression model",
    )
    parser.add_argument(
        '--model',
        default='final_model/final_lr_model.joblib',
        help='Path to the trained model pipeline file',
    )

    subparsers = parser.add_subparsers(dest='command', required=True)

    # info
    p_info = subparsers.add_parser(
        'info', help='Show model information and summary')
    p_info.set_defaults(func=cmd_info)

    # features
    p_feat = subparsers.add_parser(
        'features', help='List expected input features for prediction')
    p_feat.add_argument(
        '--output', nargs='?', const='', help='Optional path to save feature list CSV (if not provided, prints to console)')
    p_feat.set_defaults(func=cmd_features)

    # predict
    p_pred = subparsers.add_parser(
        'predict', help='Run batch predictions on an input CSV')
    p_pred.add_argument('--input', required=True,
                        help='Input CSV path containing base features')
    p_pred.add_argument('--output', help='Output CSV path for predictions')
    p_pred.add_argument('--threshold', type=float, default=0.5,
                        help='Decision threshold for positive label (default 0.5)')
    p_pred.add_argument(
        '--id-column', help='Optional ID column name to include in output')
    p_pred.set_defaults(func=cmd_predict)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
