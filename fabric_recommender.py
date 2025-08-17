import argparse
import json
import os
import re
import sys
import unicodedata
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


REMOTE_DATA_URL = "https://raw.githubusercontent.com/V09-ar/Materials-Dataset-/main/Dataset.xlsx"
LOCAL_DATA_PATH_DEFAULT = "/workspace/Dataset.xlsx"

# Canonical column names as expected in the dissertation
CANONICAL_COLUMNS = {
	"target": "Comfort Score (1–10)",
	"moisture": "Moisture Regain (%)",
	"water": "Water Absorption (g/m²)",
	"drying": "Drying Time (min)",
	"thermal": "Thermal Conductivity (W/m·K)",
	"fabric_type": "Fabric Type",
	"reference": "Source / Literature Reference",
}


def _normalize_column_name(name: str) -> str:
	text = unicodedata.normalize("NFKD", str(name))
	text = text.encode("ascii", "ignore").decode("ascii")
	text = text.lower()
	text = re.sub(r"[^a-z0-9]+", "", text)
	return text


def _resolve_column(df: pd.DataFrame, expected: str, synonyms: Optional[List[str]] = None) -> Optional[str]:
	norm_to_actual = {_normalize_column_name(col): col for col in df.columns}
	expected_norm = _normalize_column_name(expected)
	if expected_norm in norm_to_actual:
		return norm_to_actual[expected_norm]
	if synonyms:
		for s in synonyms:
			s_norm = _normalize_column_name(s)
			if s_norm in norm_to_actual:
				return norm_to_actual[s_norm]
	return None


def _find_required_columns(df: pd.DataFrame) -> Tuple[Dict[str, str], List[str]]:
	missing: List[str] = []
	resolved: Dict[str, str] = {}

	# Define light synonyms to be robust to symbol variations in spreadsheets
	synonyms_map: Dict[str, List[str]] = {
		"target": ["Comfort Score (1-10)", "Comfort Score"],
		"water": ["Water Absorption (g/m2)", "Water Absorption (g/m^2)", "WaterAbsorption"],
		"thermal": ["Thermal Conductivity (W/mK)", "Thermal Conductivity (W/m*K)", "ThermalConductivity"],
	}

	for key, expected in CANONICAL_COLUMNS.items():
		col = _resolve_column(df, expected, synonyms_map.get(key))
		if col is None:
			missing.append(expected)
		else:
			resolved[key] = col

	return resolved, missing


def load_dataset(local_path: Optional[str] = None, remote_url: Optional[str] = None) -> pd.DataFrame:
	path = local_path or LOCAL_DATA_PATH_DEFAULT
	url = remote_url or REMOTE_DATA_URL

	if path and os.path.exists(path):
		df = pd.read_excel(path)
	else:
		df = pd.read_excel(url)
	return df


def coerce_numeric(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
	for c in columns:
		df[c] = pd.to_numeric(df[c], errors="coerce")
	return df


def train_model(df: pd.DataFrame, feature_cols: List[str], target_col: str, random_state: int = 42) -> Tuple[Pipeline, Dict[str, float], Dict[str, float]]:
	X = df[feature_cols].copy()
	y = df[target_col].copy()

	# Drop missing rows on X or y
	mask = X.notna().all(axis=1) & y.notna()
	X = X[mask]
	y = y[mask]

	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=0.25, random_state=random_state
	)

	# Tree-based models do not require scaling, but keeping scaler ensures consistent handling of user input
	pipeline = Pipeline(
		steps=[
			("scaler", StandardScaler()),
			("model", RandomForestRegressor(n_estimators=500, random_state=random_state, n_jobs=-1)),
		]
	)

	# Cross-validation for dissertation-level reporting
	cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
	scoring = {
		"r2": "r2",
		"neg_rmse": "neg_root_mean_squared_error",
		"neg_mae": "neg_mean_absolute_error",
	}
	cv_results = cross_validate(pipeline, X, y, scoring=scoring, cv=cv, n_jobs=-1)

	pipeline.fit(X_train, y_train)
	preds_train = pipeline.predict(X_train)
	preds_test = pipeline.predict(X_test)

	metrics_train = {
		"r2": float(r2_score(y_train, preds_train)),
		"rmse": float(np.sqrt(mean_squared_error(y_train, preds_train))),
		"mae": float(mean_absolute_error(y_train, preds_train)),
	}
	metrics_test = {
		"r2": float(r2_score(y_test, preds_test)),
		"rmse": float(np.sqrt(mean_squared_error(y_test, preds_test))),
		"mae": float(mean_absolute_error(y_test, preds_test)),
	}

	# Add CV means for reference
	metrics_test.update({
		"cv_r2_mean": float(np.mean(cv_results["test_r2"])),
		"cv_rmse_mean": float(-np.mean(cv_results["test_neg_rmse"])),
		"cv_mae_mean": float(-np.mean(cv_results["test_neg_mae"])),
	})

	return pipeline, metrics_train, metrics_test


def map_user_profile_to_features(environment: str, sweating: str, activity_level: str) -> Dict[str, float]:
	# Defaults
	moisture_regain = 10.0
	water_absorption = 1000.0
	drying_time = 90.0
	thermal_conductivity = 0.04

	# Environment adjustments
	env = (environment or "").strip().lower()
	if env == "hot":
		drying_time = 60.0
		thermal_conductivity = 0.05
	elif env == "humid":
		water_absorption = 1300.0
		drying_time = 100.0
	elif env == "cold":
		thermal_conductivity = 0.03
		drying_time = 110.0
	elif env == "mild":
		drying_time = 80.0

	# Sweating adjustments
	sweat = (sweating or "").strip().lower()
	if sweat == "low":
		moisture_regain = 6.0
	elif sweat == "medium":
		moisture_regain = 10.0
	elif sweat == "high":
		moisture_regain = 15.0
		water_absorption += 200.0

	# Activity adjustments
	activity = (activity_level or "").strip().lower()
	if activity == "rest":
		drying_time += 10.0
	elif activity == "moderate":
		drying_time -= 5.0
	elif activity == "intense":
		drying_time -= 15.0
		water_absorption += 100.0

	return {
		CANONICAL_COLUMNS["moisture"]: moisture_regain,
		CANONICAL_COLUMNS["water"]: water_absorption,
		CANONICAL_COLUMNS["drying"]: drying_time,
		CANONICAL_COLUMNS["thermal"]: thermal_conductivity,
	}


def _minmax(x: np.ndarray) -> np.ndarray:
	if x.size == 0:
		return x
	mn, mx = np.min(x), np.max(x)
	if mx - mn < 1e-12:
		return np.zeros_like(x)
	return (x - mn) / (mx - mn)


def recommend_fabrics(
	pipeline: Pipeline,
	df: pd.DataFrame,
	feature_cols: List[str],
	target_col: str,
	user_profile: Dict[str, float],
	top_k: int = 3,
) -> pd.DataFrame:
	X_all = df[feature_cols].copy()

	# Predict for dataset rows and user profile
	dataset_pred = pipeline.predict(X_all)
	user_X = pd.DataFrame([user_profile], columns=feature_cols)
	user_pred = float(pipeline.predict(user_X)[0])

	# Distances in feature space (scaled) if scaler is present
	scaler = pipeline.named_steps.get("scaler")
	if scaler is not None:
		X_all_scaled = scaler.transform(X_all)
		user_scaled = scaler.transform(user_X)
		feature_dist = np.linalg.norm(X_all_scaled - user_scaled, axis=1)
	else:
		feature_dist = np.linalg.norm(X_all.values - user_X.values, axis=1)

	pred_diff = np.abs(dataset_pred - user_pred)

	# Rank strictly by predicted comfort difference to align with dissertation spec
	rank_score = pred_diff

	result = df.copy()
	result["Predicted_Comfort"] = dataset_pred
	result["User_Predicted_Comfort"] = user_pred
	result["Predicted_Difference"] = pred_diff
	result["Feature_Distance"] = feature_dist
	result["RankScore"] = rank_score

	result_sorted = result.sort_values(by=["RankScore"]).head(max(1, int(top_k)))
	return result_sorted


def save_artifacts(pipeline: Pipeline, feature_cols: List[str], metrics: Dict[str, Dict[str, float]], out_dir: str = "artifacts") -> None:
	os.makedirs(out_dir, exist_ok=True)
	dump(pipeline, os.path.join(out_dir, "fabric_recommender.joblib"))
	with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as f:
		json.dump({
			"feature_columns": feature_cols,
			"metrics": metrics,
		}, f, indent=2)


def print_recommendations(df_rec: pd.DataFrame, feature_cols: List[str], fabric_col: Optional[str], reference_col: Optional[str]) -> None:
	from tabulate import tabulate

	cols_to_show: List[str] = []
	if fabric_col and fabric_col in df_rec.columns:
		cols_to_show.append(fabric_col)
	cols_to_show += ["Predicted_Comfort", "User_Predicted_Comfort", "Predicted_Difference"]
	for c in feature_cols:
		if c in df_rec.columns:
			cols_to_show.append(c)
	if reference_col and reference_col in df_rec.columns:
		cols_to_show.append(reference_col)

	preview = df_rec[cols_to_show].copy()
	if "Predicted_Comfort" in preview:
		preview["Predicted_Comfort"] = preview["Predicted_Comfort"].round(2)
	if "User_Predicted_Comfort" in preview:
		preview["User_Predicted_Comfort"] = preview["User_Predicted_Comfort"].round(2)
	if "Predicted_Difference" in preview:
		preview["Predicted_Difference"] = preview["Predicted_Difference"].round(2)

	print("\nRecommended fabric(s):")
	print(tabulate(preview, headers="keys", tablefmt="github", showindex=False))


def main(argv: Optional[List[str]] = None) -> int:
	parser = argparse.ArgumentParser(description="Fabric Comfort Recommender (dissertation-ready)")
	parser.add_argument("--env", dest="environment", type=str, choices=["hot", "humid", "cold", "mild"], help="Environment: hot | humid | cold | mild")
	parser.add_argument("--sweat", dest="sweating", type=str, choices=["low", "medium", "high"], help="Sweating level: low | medium | high")
	parser.add_argument("--activity", dest="activity", type=str, choices=["rest", "moderate", "intense"], help="Activity level: rest | moderate | intense")
	parser.add_argument("--top_k", dest="top_k", type=int, default=3, help="Number of recommendations to return")
	parser.add_argument("--data", dest="data_path", type=str, default=LOCAL_DATA_PATH_DEFAULT, help="Local dataset path (Excel)")
	parser.add_argument("--remote_url", dest="remote_url", type=str, default=REMOTE_DATA_URL, help="Remote dataset URL fallback")
	parser.add_argument("--no_save", dest="no_save", action="store_true", help="Do not save model artifacts")
	args = parser.parse_args(argv)

	# Load dataset
	df = load_dataset(args.data_path, args.remote_url)

	resolved_cols, missing = _find_required_columns(df)
	if missing:
		raise ValueError(
			"Missing required columns in dataset: " + ", ".join(missing) + "\nAvailable columns: " + ", ".join(map(str, df.columns))
		)

	feature_cols = [
		resolved_cols["moisture"],
		resolved_cols["water"],
		resolved_cols["drying"],
		resolved_cols["thermal"],
	]
	target_col = resolved_cols["target"]
	fabric_col = resolved_cols.get("fabric_type")
	reference_col = resolved_cols.get("reference")

	# Ensure numeric types for features and target
	df = coerce_numeric(df, feature_cols + [target_col])
	# Drop rows with missing essential values
	df = df.dropna(subset=feature_cols + [target_col]).reset_index(drop=True)
	if df.empty:
		raise ValueError("Dataset is empty after cleaning required columns. Please check data.")

	# Train and evaluate
	pipeline, metrics_train, metrics_test = train_model(df, feature_cols, target_col)

	metrics = {"train": metrics_train, "test": metrics_test}
	print("\nModel performance:")
	print(json.dumps(metrics, indent=2))

	# Save artifacts
	if not args.no_save:
		save_artifacts(pipeline, feature_cols, metrics)
		print("\nArtifacts saved to ./artifacts")

	# Gather user profile (CLI or interactive fallback)
	environment = args.environment
	sweating = args.sweating
	activity = args.activity
	if not (environment and sweating and activity):
		print("\nInteractive mode (no CLI profile provided).")
		environment = (input("Environment (hot / humid / cold / mild): ").strip().lower() or "mild")
		sweating = (input("Sweating level (low / medium / high): ").strip().lower() or "medium")
		activity = (input("Activity level (rest / moderate / intense): ").strip().lower() or "moderate")

	user_profile = map_user_profile_to_features(environment, sweating, activity)

	recs = recommend_fabrics(
		pipeline=pipeline,
		df=df,
		feature_cols=feature_cols,
		target_col=target_col,
		user_profile=user_profile,
		top_k=args.top_k,
	)

	print_recommendations(recs, feature_cols, fabric_col, reference_col)
	return 0


if __name__ == "__main__":
	sys.exit(main())