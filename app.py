from flask import Flask, render_template, request, send_file, session
import joblib
import numpy as np
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, ListFlowable, ListItem
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import A4
from datetime import datetime
import os

app = Flask(__name__)
app.secret_key = "supersecretkey"

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")


# ---------------- HOME ----------------
@app.route("/")
def home():
    return render_template("home.html")


# ---------------- CLINICAL FORM PAGE ----------------
@app.route("/screening")
def screening():
    return render_template("screening.html")


# ---------------- CLINICAL PREDICTION ----------------
@app.route("/predict", methods=["POST"])
def predict():

    features = [
        float(request.form["pregnancies"]),
        float(request.form["glucose"]),
        float(request.form["bloodpressure"]),
        float(request.form["skinthickness"]),
        float(request.form["insulin"]),
        float(request.form["bmi"]),
        float(request.form["dpf"]),
        float(request.form["age"]),
    ]

    scaled = scaler.transform([features])
    prob = model.predict_proba(scaled)[0][1]
    probability = round(prob * 100, 2)

    names = [
        "Pregnancies", "Glucose", "Blood Pressure", "Skin Thickness",
        "Insulin", "BMI", "Diabetes Pedigree", "Age"
    ]
    feature_data = []

    scaled_values = scaled[0]   # use scaled patient values



# If Tree Model (RandomForest / DecisionTree)
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_

    for i in range(len(features)):
        contribution = scaled_values[i] * importances[i]
        feature_data.append(
            (names[i], round(abs(float(contribution)), 4))
        )


# Sort highest impact first
    feature_data.sort(key=lambda x: x[1], reverse=True)

    # -------- RISK + RECOMMENDATIONS --------
    if probability < 35:
        risk = "Low Risk"
        color = "green"
        diet_plan = [
            "Whole grains and leafy vegetables",
            "Lean proteins like fish and dal",
            "Limit sugar intake"
        ]
        exercise_plan = [
            "Brisk walking 30 minutes daily",
            "Light yoga or stretching"
        ]

    elif probability < 65:
        risk = "Moderate Risk"
        color = "orange"
        diet_plan = [
            "Low glycemic index foods",
            "High fiber diet",
            "Avoid sugary drinks"
        ]
        exercise_plan = [
            "Cardio 5 days per week",
            "Strength training 3 days per week"
        ]

    else:
        risk = "High Risk"
        color = "red"
        diet_plan = [
            "Strict diabetic diet plan",
            "Avoid processed carbs",
            "Monitor calorie intake"
        ]
        exercise_plan = [
            "Daily supervised exercise",
            "Regular walking + resistance training"
        ]

    # SAVE TO SESSION
    session["type"] = "clinical"
    session["probability"] = probability
    session["risk"] = risk
    session["features"] = feature_data
    session["diet_plan"] = diet_plan
    session["exercise_plan"] = exercise_plan

    return render_template(
        "result.html",
        probability=probability,
        risk=risk,
        color=color,
        features=feature_data,
        diet_plan=diet_plan,
        exercise_plan=exercise_plan
    )


# ---------------- SYMPTOM PAGE ----------------
@app.route("/symptom")
def symptom():
    return render_template("symptom.html")


# ---------------- SYMPTOM PREDICTION ----------------
@app.route("/predict_symptom", methods=["POST"])
def predict_symptom():

    score = 0
    for key in request.form:
        if request.form.get(key) == "yes":
            score += 1

    probability = int((score / 5) * 100)

    if probability >= 70:
        risk = "High Risk"
        color = "red"
        doctor_advice = "Strongly recommended to consult a doctor immediately."
        precautions = [
            "Reduce sugar intake immediately",
            "Start daily physical activity",
            "Monitor blood sugar levels",
            "Schedule medical consultation"
        ]
        diet_plan = [
            "Strict diabetic diet",
            "Avoid refined sugar",
            "High fiber vegetables"
        ]
        exercise_plan = [
            "30 mins brisk walking daily",
            "Doctor guided fitness plan"
        ]

    elif probability >= 40:
        risk = "Moderate Risk"
        color = "orange"
        doctor_advice = "Consult a doctor if symptoms persist."
        precautions = [
            "Improve diet quality",
            "Exercise 30 minutes daily",
            "Avoid processed foods",
            "Monitor symptoms regularly"
        ]
        diet_plan = [
            "Low glycemic foods",
            "Avoid sugary drinks",
            "Increase fiber intake"
        ]
        exercise_plan = [
            "Cardio 5 days/week",
            "Light strength training"
        ]

    else:
        risk = "Low Risk"
        color = "green"
        doctor_advice = "Maintain healthy lifestyle and annual checkup."
        precautions = [
            "Maintain balanced diet",
            "Stay physically active",
            "Regular health screening",
            "Avoid excessive sugar"
        ]
        diet_plan = [
            "Whole grains",
            "Leafy vegetables",
            "Lean proteins"
        ]
        exercise_plan = [
            "Brisk walking 30 mins/day",
            "Stretching exercises"
        ]

    # SAVE TO SESSION
    session["type"] = "symptom"
    session["probability"] = probability
    session["risk"] = risk
    session["precautions"] = precautions
    session["doctor_advice"] = doctor_advice
    session["diet_plan"] = diet_plan
    session["exercise_plan"] = exercise_plan

    return render_template(
        "symptom_result.html",
        probability=probability,
        risk_level=risk,
        color=color,
        precautions=precautions,
        doctor_advice=doctor_advice,
        diet_plan=diet_plan,
        exercise_plan=exercise_plan
    )


# ---------------- DOWNLOAD REPORT ----------------
@app.route("/download_report")
def download_report():

    if "probability" not in session:
        return "Please complete screening first."

    file_path = "diabetes_report.pdf"
    doc = SimpleDocTemplate(file_path, pagesize=A4)
    elements = []
    styles = getSampleStyleSheet()

    elements.append(Paragraph("Smart Diabetes Screening Report", styles["Heading1"]))
    elements.append(Spacer(1, 0.3 * inch))

    elements.append(Paragraph(
        f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        styles["Normal"]
    ))
    elements.append(Spacer(1, 0.3 * inch))

    elements.append(Paragraph(f"Risk Level: {session['risk']}", styles["Heading2"]))
    elements.append(Spacer(1, 0.2 * inch))

    elements.append(Paragraph(f"Probability: {session['probability']}%", styles["Normal"]))
    elements.append(Spacer(1, 0.3 * inch))

    # Clinical Graph
    if session.get("type") == "clinical":

        feature_data = session.get("features")
        features = [f[0] for f in feature_data]
        values = [f[1] for f in feature_data]

        graph_path = "feature_graph.png"

        plt.figure()
        plt.bar(features, values)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(graph_path)
        plt.close()

        elements.append(Paragraph("Feature Impact Analysis", styles["Heading2"]))
        elements.append(Spacer(1, 0.3 * inch))
        elements.append(Image(graph_path, width=5 * inch, height=3 * inch))
        elements.append(Spacer(1, 0.3 * inch))

    # Symptom Advice
    if session.get("type") == "symptom":

        elements.append(Paragraph("Doctor Advice:", styles["Heading2"]))
        elements.append(Spacer(1, 0.2 * inch))
        elements.append(Paragraph(session["doctor_advice"], styles["Normal"]))
        elements.append(Spacer(1, 0.3 * inch))

        elements.append(Paragraph("Precautions:", styles["Heading2"]))
        elements.append(Spacer(1, 0.2 * inch))

        elements.append(ListFlowable(
            [ListItem(Paragraph(item, styles["Normal"])) for item in session["precautions"]]
        ))

    # Diet Plan
    elements.append(Spacer(1, 0.3 * inch))
    elements.append(Paragraph("Diet Plan:", styles["Heading2"]))
    elements.append(Spacer(1, 0.2 * inch))
    elements.append(ListFlowable(
        [ListItem(Paragraph(item, styles["Normal"])) for item in session["diet_plan"]]
    ))

    # Exercise Plan
    elements.append(Spacer(1, 0.3 * inch))
    elements.append(Paragraph("Exercise Plan:", styles["Heading2"]))
    elements.append(Spacer(1, 0.2 * inch))
    elements.append(ListFlowable(
        [ListItem(Paragraph(item, styles["Normal"])) for item in session["exercise_plan"]]
    ))

    doc.build(elements)

    return send_file(file_path, as_attachment=True)


# ---------------- RUN APP ----------------
if __name__ == "__main__":
    app.run(debug=True)