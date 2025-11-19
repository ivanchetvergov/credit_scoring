# scripts/generate_features_config.py
"""
CLI wrapper to generate/update feature lists for src/configs/raw_features_config.py.
Uses the project's helper `src.features.data_helpers.inspect_and_suggest` when available.

Usage:
    python scripts/generate_features_config.py [--apply] [--show-config]

Options:
    --apply        Append the suggested patch to configs/raw_features_config.py
    --show-config  Print resulting NUMERICAL_FEATURES / CATEGORICAL_FEATURES / BIN_CATEGORICAL_FEATURES
"""
from pathlib import Path
import argparse
import importlib
import sys

CONFIG_PATH = Path('configs/raw_features_config.py')

def fallback_generate_and_print(feature_store_path: Path):
    """Fallback: read parquet and print simple generated lists (basic dtype heuristics)."""
    import pandas as pd
    try:
        df = pd.read_parquet(feature_store_path)
    except Exception as e:
        print(f"Cannot read feature store at {feature_store_path}: {e}")
        raise

    EXCLUDED_COLS = ['SK_ID_CURR', 'TARGET']
    X = df.drop(columns=EXCLUDED_COLS, errors='ignore')

    numerical = X.select_dtypes(include=['number']).columns.tolist()
    categorical = X.select_dtypes(include=['object', 'category']).columns.tolist()

    # detect binary-like in numeric
    bin_cols = []
    numeric_remaining = numerical.copy()
    for c in numerical:
        vals = X[c].dropna().unique()
        if set(vals).issubset({0, 1}):
            bin_cols.append(c)
            numeric_remaining.remove(c)

    numerical = numeric_remaining

    print('=' * 80)
    print('Suggested feature lists:')
    print('\nNUMERICAL_FEATURES = [')
    for c in sorted(numerical):
        print(f"    '{c}',")
    print(']')

    print('\nCATEGORICAL_FEATURES = [')
    for c in sorted(categorical):
        print(f"    '{c}',")
    print(']')

    print('\nBIN_CATEGORICAL_FEATURES = [')
    for c in sorted(bin_cols):
        print(f"    '{c}',")
    print(']')
    print('=' * 80)


def main():
    parser = argparse.ArgumentParser(description='Generate or update feature lists in configs/raw_features_config.py')
    parser.add_argument('--apply', action='store_true', help='Append suggested patch to configs/raw_features_config.py')
    parser.add_argument('--show-config', action='store_true', help='After applying (or not), print final lists from configs/raw_features_config.py')
    parser.add_argument('--feature-store', type=Path, default=None, help='Optional path to feature_store.parquet (overrides src.config)')
    args = parser.parse_args()

    # Try to use helper inspect_and_suggest if available
    try:
        from src.features import data_helpers
        # prefer the helper's default feature store path; allow override
        if args.feature_store is not None:
            feature_store_arg = args.feature_store
        else:
            try:
                from src.config import FEATURE_STORE_PATH
                feature_store_arg = Path(FEATURE_STORE_PATH)
            except Exception:
                feature_store_arg = None

        # call helper (it will print patch preview). It accepts apply_patch param.
        print('Using src.features.data_helpers.inspect_and_suggest to build suggestion...')
        report = data_helpers.inspect_and_suggest(
            pipeline_path=Path('models/pipeline.joblib'),
            feature_store_path=feature_store_arg or Path('data/processed/feature_store.parquet'),
            config_module_name='configs.raw_features_config',
            apply_patch=args.apply
        )

    except Exception as e:
        # If data_helpers not available or failing, fallback to simple generator
        print('Helper data_helpers not available or failed:', e)
        # determine feature store path
        feature_store = args.feature_store or Path('data/processed/feature_store.parquet')
        fallback_generate_and_print(feature_store)
        print('\nTo enable the richer flow, ensure src.features.data_helpers.inspect_and_suggest is present and importable.')
        sys.exit(0)

    # Optionally show final lists from config
    if args.show_config:
        try:
            cfg = importlib.reload(importlib.import_module('configs.raw_features_config'))
            print('\nFinal lists from configs/raw_features_config.py:')
            for attr in ('NUMERICAL_FEATURES', 'CATEGORICAL_FEATURES', 'BIN_CATEGORICAL_FEATURES'):
                if hasattr(cfg, attr):
                    print(f"\n{attr} = [ (count={len(getattr(cfg, attr))}) ]")
                    for c in getattr(cfg, attr):
                        print('   ', c)
                else:
                    print(f"\n{attr} not found in configs/raw_features_config.py")
        except Exception as exc:
            print('Could not import configs.raw_features_config:', exc)


if __name__ == '__main__':
    main()
