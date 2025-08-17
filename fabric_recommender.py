import os
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.preprocessing import StandardScaler


DATASET_LOCAL_PATH = "/workspace/Dataset.xlsx"
DATASET_REMOTE_URL = (
	"https://raw.githubusercontent.com/V09-ar/Materials-Dataset-/main/Dataset.xlsx"
)

# Column names expected by the dissertation dataset
FEATURE_COLUMNS = [
	"Moisture Regain (%)",
	"Water Absorption (g/m²)",
	"Drying Time (min)",
	"Thermal Conductivity (W/m·K)",
]
TARGET_COLUMN = "Comfort Score (1–10)"
FABRIC_COLUMN = "Fabric Type"
REFERENCE_COLUMN = "Source / Literature Reference"


def load_dataset() -> pd.DataFrame:
	if os.path.exists(DATASET_LOCAL_PATH):
		df = pd.read_excel(DATASET_LOCAL_PATH)
	else:
		df = pd.read_excel(DATASET_REMOTE_URL)
	# Basic validation
	missing = [c for c in FEATURE_COLUMNS + [TARGET_COLUMN, FABRIC_COLUMN, REFERENCE_COLUMN] if c not in df.columns]
	if missing:
		raise ValueError(f"Dataset is missing required columns: {missing}")
	return df


def train_model(df: pd.DataFrame) -> Tuple[StandardScaler, RandomForestRegressor]:
	X = df[FEATURE_COLUMNS].copy()
	y = df[TARGET_COLUMN].astype(float).copy()

	# Scale numeric features (kept for consistency with the original approach)
	scaler = StandardScaler()
	X_scaled = scaler.fit_transform(X)

	# RandomForest is robust to feature scaling, but we keep scaling for parity with dissertation code
	model = RandomForestRegressor(random_state=42, n_estimators=300)
	model.fit(X_scaled, y)
	return scaler, model


def evaluate_model_cv(df: pd.DataFrame) -> float:
	X = df[FEATURE_COLUMNS].copy()
	y = df[TARGET_COLUMN].astype(float).copy()
	scaler = StandardScaler()
	X_scaled = scaler.fit_transform(X)
	model = RandomForestRegressor(random_state=42, n_estimators=300)
	cv = LeaveOneOut()
	# R^2 is undefined for 1-sample test folds, so use MAE
	scores = cross_val_score(model, X_scaled, y, cv=cv, scoring="neg_mean_absolute_error")
	return float(-np.mean(scores))


def _clamp(value: float, low: float, high: float) -> float:
	return max(low, min(high, value))


def map_preferences_to_features(environment: str, sweating: str, activity_level: str) -> np.ndarray:
	# Sensible defaults (mid-range) aligned with dataset magnitudes
	moisture_regain = 10.0
	water_absorption = 1200.0
	drying_time = 90.0
	thermal_conductivity = 0.04

	# Environment adjustments
	environment = (environment or "").strip().lower()
	if environment == "hot":
		# Prefer faster drying and slightly higher conductivity for heat transfer
		drying_time = 60.0
		thermal_conductivity = 0.05
	elif environment == "humid":
		# Higher absorption but expect longer drying
		water_absorption = 1300.0
		drying_time = 100.0
	elif environment == "cold":
		# Lower conductivity (more insulating)
		thermal_conductivity = 0.03
		drying_time = 110.0
	elif environment == "mild":
		drying_time = 80.0

	# Sweating level adjustments
	sweating = (sweating or "").strip().lower()
	if sweating == "low":
		moisture_regain = 6.0
	elif sweating == "medium":
		moisture_regain = 10.0
	elif sweating == "high":
		moisture_regain = 15.0
		water_absorption += 200.0

	# Activity level adjustments
	activity_level = (activity_level or "").strip().lower()
	if activity_level == "rest":
		drying_time += 10.0
	elif activity_level == "moderate":
		drying_time -= 5.0
	elif activity_level == "intense":
		drying_time -= 15.0
		water_absorption += 100.0

	# Clamp to reasonable bounds based on dataset ranges
	# These ranges reflect and slightly extend the observed dataset values
	moisture_regain = _clamp(moisture_regain, 5.0, 20.0)
	water_absorption = _clamp(water_absorption, 800.0, 1600.0)
	drying_time = _clamp(drying_time, 50.0, 140.0)
	thermal_conductivity = _clamp(thermal_conductivity, 0.02, 0.08)

	return np.array([[moisture_regain, water_absorption, drying_time, thermal_conductivity]], dtype=float)


def recommend_fabrics(
	environment: str,
	sweating: str,
	activity_level: str,
	top_k: int = 3,
) -> None:
	"""Print top-k recommended fabrics based on predicted comfort proximity.

	We compute:
	1) Predicted comfort for the user's mapped features
	2) Predicted comfort for each fabric in the dataset
	3) Rank fabrics by absolute difference between their predicted comfort and the user's predicted comfort
	"""
	# Load and train
	df = load_dataset()
	scaler, model = train_model(df)

	# Map preferences to feature vector
	user_features = map_preferences_to_features(environment, sweating, activity_level)
	user_features_df = pd.DataFrame(user_features, columns=FEATURE_COLUMNS)
	user_features_scaled = scaler.transform(user_features_df)

	# Predict user's comfort score
	user_pred_score = float(model.predict(user_features_scaled)[0])

	# Predict comfort for each fabric using its own features
	fabric_features_scaled = scaler.transform(df[FEATURE_COLUMNS])
	df_predictions = model.predict(fabric_features_scaled)

	# Compute differences and sort
	result_df = df[[FABRIC_COLUMN, REFERENCE_COLUMN] + FEATURE_COLUMNS + [TARGET_COLUMN]].copy()
	result_df["Predicted Comfort"] = df_predictions
	result_df["|Diff to User|"] = np.abs(result_df["Predicted Comfort"] - user_pred_score)
	result_df = result_df.sort_values(by="|Diff to User|").reset_index(drop=True)

	# Display results
	print("\n🔹 Recommended Fabrics for You 🔹")
	print(f"Predicted user comfort: {user_pred_score:.2f} / 10\n")

	top_k = max(1, min(int(top_k), len(result_df)))
	for rank in range(top_k):
		row = result_df.iloc[rank]
		print(f"#{rank+1}: {row[FABRIC_COLUMN]}")
		print(f"  Predicted Comfort   : {row['Predicted Comfort']:.2f} / 10 (diff {row['|Diff to User|']:.2f})")
		print(f"  Actual Comfort      : {row[TARGET_COLUMN]} / 10")
		print(f"  Moisture Regain     : {row['Moisture Regain (%)']} %")
		print(f"  Water Absorption    : {row['Water Absorption (g/m²)']} g/m²")
		print(f"  Drying Time         : {row['Drying Time (min)']} min")
		print(f"  Thermal Conductivity: {row['Thermal Conductivity (W/m·K)']} W/m·K")
		print(f"  Reference           : {row[REFERENCE_COLUMN]}\n")


def _interactive_main():
	print("\n✨ Welcome to the Fabric Recommender ✨")
	env = input("🌍 Enter environment (hot / humid / cold / mild): ").strip().lower()
	sweat = input("💧 Enter sweating level (low / medium / high): ").strip().lower()
	activity = input("🏃 Enter activity level (rest / moderate / intense): ").strip().lower()
	try:
		recommend_fabrics(env, sweat, activity, top_k=3)
	except Exception as exc:
		print(f"Error: {exc}")
		sys.exit(1)


if __name__ == "__main__":
	# If the script is run directly without arguments, use interactive mode
	# Optional CLI args: env sweat activity [top_k]
	if len(sys.argv) >= 4:
		env = sys.argv[1]
		sweat = sys.argv[2]
		activity = sys.argv[3]
		top_k = int(sys.argv[4]) if len(sys.argv) >= 5 else 3
		recommend_fabrics(env, sweat, activity, top_k=top_k)
	else:
		_interactive_main()