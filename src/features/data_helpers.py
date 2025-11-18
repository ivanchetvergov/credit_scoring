# src/features/data_helpers.py
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
from src.config import FEATURE_STORE, PIPELINE_PATH, CONFIG_PATH

# ========================== #
 #  вспомогательные функции #
# ========================== #

def get_pipeline_raw_inputs(pipeline) -> List[str]:
    pre = pipeline.named_steps.get("preprocessor", None)
    if pre is None:
        return []
    cols = []
    if hasattr(pre, "transformers_"):
        for name, trans, cols_spec in pre.transformers_:
            if name == "remainder":
                continue
            # cols_spec can be slice, list, array, string
            if isinstance(cols_spec, (list, tuple, np.ndarray)):
                cols.extend(list(cols_spec))
            elif isinstance(cols_spec, str):
                cols.append(cols_spec)
            else:
                try:
                    cols.extend(list(cols_spec))
                except Exception:
                    pass
    return list(dict.fromkeys(cols))

# compare feature store vs pipeline raw inputs vs config lists
def analyze_feature_coverage(feature_store_path: Path, pipeline=None, config_module=None) -> Dict[str, Any]:
    df = pd.read_parquet(feature_store_path)
    store_cols = df.columns.tolist()
    # remove control columns
    store_cols_clean = [c for c in store_cols if c not in ("SK_ID_CURR", "TARGET")]

    used_raw = get_pipeline_raw_inputs(pipeline) if pipeline is not None else []
    config_raw = []
    if config_module is not None:
        config_raw = []
        for attr in ("NUMERICAL_FEATURES", "CATEGORICAL_FEATURES", "BIN_CATEGORICAL_FEATURES"):
            if hasattr(config_module, attr):
                config_raw += getattr(config_module, attr)
        config_raw = list(dict.fromkeys(config_raw))

    missing_in_pipeline = sorted(set(store_cols_clean) - set(used_raw))
    missing_in_config = sorted(set(store_cols_clean) - set(config_raw))
    extra_in_config = sorted(set(config_raw) - set(store_cols_clean))

    return {
        "store_cols_count": len(store_cols_clean),
        "used_raw": used_raw,
        "config_raw": config_raw,
        "missing_in_pipeline": missing_in_pipeline,
        "missing_in_config": missing_in_config,
        "extra_in_config": extra_in_config,
        "df_sample_dtypes": df.dtypes[missing_in_config].to_dict() if missing_in_config else {}
    }

# prepare patch text to append to configs/raw_features_config.py
def suggest_config_patch(feature_store_path: Path, missing_features: List[str]) -> str:
    if not missing_features:
        return "# No missing features to add"
    df = pd.read_parquet(feature_store_path)
    num_add = [c for c in missing_features if pd.api.types.is_numeric_dtype(df[c])]
    cat_add = [c for c in missing_features if not pd.api.types.is_numeric_dtype(df[c])]
    # small heuristic for binary features
    bin_add = [c for c in cat_add if set(df[c].dropna().unique()).issubset({0,1})]
    # remove bin_add from cat_add
    cat_add = [c for c in cat_add if c not in bin_add]

    parts = []
    if num_add:
        parts.append("NUMERICAL_FEATURES += [\n    " + ",\n    ".join(f"'{c}'" for c in num_add) + "\n]\n")
    if cat_add:
        parts.append("CATEGORICAL_FEATURES += [\n    " + ",\n    ".join(f"'{c}'" for c in cat_add) + "\n]\n")
    if bin_add:
        parts.append("BIN_CATEGORICAL_FEATURES += [\n    " + ",\n    ".join(f"'{c}'" for c in bin_add) + "\n]\n")
    patch = "\n# --- Auto-suggested additions (generated) ---\n" + "\n".join(parts)
    return patch

# optionally apply patch (append) to config file
def apply_patch_to_config(config_path: Path, patch_text: str, dry_run: bool = True) -> None:
    if dry_run:
        print("Dry run - patch preview:\n")
        print(patch_text)
        return
    with open(config_path, "a", encoding="utf-8") as f:
        f.write("\n" + patch_text)
    print(f"Appended patch to {config_path}")

# retrain pipeline end-to-end (loads config lists at runtime)
def retrain_pipeline(feature_store_path: Path,
                     config_module_path: str = "configs.raw_features_config",
                     pipeline_builder_module: str = "src.pipelines.base_pipeline",
                     pipeline_creator_name: str = "create_baseline_pipeline",
                     model_kwargs: dict = None,
                     test_size: float = 0.2,
                     random_state: int = 42) -> Tuple[Any, Dict[str, float]]:
    import importlib
    # load feature store
    df = pd.read_parquet(feature_store_path)
    # import config module (reload to pick recent changes)
    cfg = importlib.reload(importlib.import_module(config_module_path))
    feature_list = []
    for attr in ("NUMERICAL_FEATURES", "CATEGORICAL_FEATURES", "BIN_CATEGORICAL_FEATURES"):
        feature_list += getattr(cfg, attr, [])
    # ensure features exist
    feature_list = [f for f in feature_list if f in df.columns]
    if "TARGET" not in df.columns:
        raise ValueError("TARGET column missing in feature store")
    X = df[feature_list]
    y = df["TARGET"]
    # split
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
    # import builder and create pipeline
    pb = importlib.import_module(pipeline_builder_module)
    model_kwargs = model_kwargs or {"random_state": random_state, "n_jobs": -1}
    from lightgbm import LGBMClassifier
    model = LGBMClassifier(**model_kwargs)
    pipeline = getattr(pb, pipeline_creator_name)(model, use_pca=False)
    # fit
    pipeline.fit(X_train, y_train)
    # quick eval
    train_score = pipeline.score(X_train, y_train)
    val_score = pipeline.score(X_val, y_val)
    # persist
    Path("models").mkdir(exist_ok=True, parents=True)
    joblib.dump(pipeline, PIPELINE_PATH)
    return pipeline, {"train_score": train_score, "val_score": val_score}

# convenience wrapper: analyze and show patch
def inspect_and_suggest(pipeline_path: Path = PIPELINE_PATH,
                        feature_store_path: Path = FEATURE_STORE,
                        config_module_name: str = "configs.raw_features_config",
                        apply_patch: bool = False) -> Dict[str, Any]:
    pipeline = None
    if pipeline_path.exists():
        pipeline = joblib.load(pipeline_path)
    import importlib
    cfg = importlib.import_module(config_module_name)
    report = analyze_feature_coverage(feature_store_path, pipeline=pipeline, config_module=cfg)
    patch = suggest_config_patch(feature_store_path, report["missing_in_config"])
    print(f"Feature store cols: {report['store_cols_count']}")
    print(f"Missing in pipeline (present in store but not fed to preprocessor): {len(report['missing_in_pipeline'])}")
    print(f"Missing in config (need to add): {len(report['missing_in_config'])}")
    print("Patch preview:")
    print(patch)
    if apply_patch:
        apply_patch_to_config(CONFIG_PATH, patch_text=patch, dry_run=False)
    return report