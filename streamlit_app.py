from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st


APP_TITLE = "Stroke Risk Prediction"
MODEL_PATH = Path("logistic_regression_stroke_model.pkl")
DATA_PATH = Path("StrokeData.csv")


@st.cache_data(show_spinner=False)
def load_training_reference():
	if not DATA_PATH.exists():
		return None, None, None

	df = pd.read_csv(DATA_PATH)

	# Match the notebook preprocessing.
	df = df.drop(columns=["id"], errors="ignore")
	df = df[df["gender"] != "Other"]

	bmi_median = df["bmi"].median()
	df["bmi"] = df["bmi"].fillna(bmi_median)

	categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
	df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

	df = add_engineered_features(df)

	if "stroke" in df.columns:
		feature_columns = [col for col in df.columns if col != "stroke"]
	else:
		feature_columns = list(df.columns)

	return feature_columns, bmi_median, df


def add_engineered_features(df):
	df = df.copy()

	df["age_group"] = pd.cut(
		df["age"],
		bins=[0, 30, 45, 60, 100],
		labels=["young", "middle", "senior", "elderly"],
	)
	df = pd.get_dummies(df, columns=["age_group"], drop_first=True)

	df["bmi_category"] = pd.cut(
		df["bmi"],
		bins=[0, 18.5, 25, 30, 100],
		labels=["underweight", "normal", "overweight", "obese"],
	)
	df = pd.get_dummies(df, columns=["bmi_category"], drop_first=True)

	df["glucose_category"] = pd.cut(
		df["avg_glucose_level"],
		bins=[0, 100, 125, 200, 1000],
		labels=["normal", "prediabetic", "diabetic", "very_high"],
	)
	df = pd.get_dummies(df, columns=["glucose_category"], drop_first=True)

	df["health_risk_score"] = df["hypertension"] + df["heart_disease"]

	return df


def build_input_frame(inputs):
	return pd.DataFrame([inputs])


def preprocess_input(df_input, bmi_median, feature_columns):
	df = df_input.copy()

	df["bmi"] = df["bmi"].fillna(bmi_median)

	categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
	df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

	df = add_engineered_features(df)

	for col in feature_columns:
		if col not in df.columns:
			df[col] = 0

	df = df[feature_columns]
	return df


@st.cache_resource(show_spinner=False)
def load_model():
	if not MODEL_PATH.exists():
		return None
	return joblib.load(MODEL_PATH)


def main():
	st.set_page_config(page_title=APP_TITLE, layout="wide")
	st.title(APP_TITLE)
	st.caption("Predict stroke risk using the trained Logistic Regression model.")

	feature_columns, bmi_median, training_df = load_training_reference()
	model = load_model()

	if feature_columns is None:
		st.error("StrokeData.csv not found. Place it next to this app.")
		st.stop()

	if model is None:
		st.error(
			"Model file not found. Run the notebook to generate logistic_regression_stroke_model.pkl."
		)
		st.stop()

	st.sidebar.header("Patient Inputs")

	with st.sidebar.form("input_form"):
		gender = st.selectbox("Gender", ["Male", "Female"])
		age = st.number_input("Age", min_value=0, max_value=100, value=45)

		hypertension = st.selectbox("Hypertension", [0, 1], format_func=lambda x: "Yes" if x else "No")
		heart_disease = st.selectbox("Heart Disease", [0, 1], format_func=lambda x: "Yes" if x else "No")

		ever_married = st.selectbox("Ever Married", ["Yes", "No"])
		work_type = st.selectbox(
			"Work Type",
			["Private", "Self-employed", "Govt_job", "children", "Never_worked"],
		)
		residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])

		avg_glucose_level = st.number_input(
			"Average Glucose Level",
			min_value=50.0,
			max_value=300.0,
			value=100.0,
			step=1.0,
		)

		bmi_unknown = st.checkbox("BMI Unknown", value=False)
		bmi = st.number_input(
			"BMI",
			min_value=10.0,
			max_value=60.0,
			value=27.0,
			step=0.1,
			disabled=bmi_unknown,
		)

		smoking_status = st.selectbox(
			"Smoking Status",
			["never smoked", "formerly smoked", "smokes", "Unknown"],
		)

		threshold = st.slider("Alert Threshold", min_value=0.1, max_value=0.9, value=0.5, step=0.05)
		submitted = st.form_submit_button("Predict")

	if submitted:
		bmi_value = np.nan if bmi_unknown else float(bmi)

		input_payload = {
			"gender": gender,
			"age": float(age),
			"hypertension": int(hypertension),
			"heart_disease": int(heart_disease),
			"ever_married": ever_married,
			"work_type": work_type,
			"Residence_type": residence_type,
			"avg_glucose_level": float(avg_glucose_level),
			"bmi": bmi_value,
			"smoking_status": smoking_status,
		}

		df_input = build_input_frame(input_payload)
		X_input = preprocess_input(df_input, bmi_median, feature_columns)

		proba = model.predict_proba(X_input.to_numpy())[0][1]
		prediction = int(proba >= threshold)

		col_left, col_right = st.columns(2)
		with col_left:
			st.subheader("Prediction")
			label = "High Risk" if prediction == 1 else "Low Risk"
			st.metric("Result", label)
			st.write(f"Stroke probability: {proba:.2%}")

		with col_right:
			st.subheader("Inputs")
			st.json(input_payload)

		st.info(
			"This tool is for educational use only and does not provide medical advice."
		)

	with st.expander("Model Details"):
		st.write(f"Features used: {len(feature_columns)}")
		if training_df is not None:
			st.write("Training data rows (after cleaning):", len(training_df))


if __name__ == "__main__":
	main()
