from flask import Flask, render_template, request
import numpy as np
import joblib
from dashboard_charts import generate_graphs

app = Flask(__name__)

# Load model and scaler
model = joblib.load("placement_model.pkl")
scaler = joblib.load("scaler.pkl")


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predict_page")
def predict_page():
    return render_template("predict.html")


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # ✅ Collect input
        attendance = float(request.form['attendance'])
        internal_marks = float(request.form['internal_marks'])
        age = float(request.form['age'])
        extra_curriculars = 1 if request.form['extra_curriculars'] == "Y" else 0
        part_time_job = 1 if request.form['part_time_job'] == "Y" else 0
        backlogs = int(request.form['backlogs'])

        # ✅ Convert input to numerical array
        input_data = np.array([[attendance, internal_marks, age,
                                extra_curriculars, part_time_job, backlogs]])

        # ✅ Scale + Predict
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1] * 100

        # ✅ Detailed Explanation
        reasons = []

        if attendance >= 75:
            reasons.append(f"✔ Good attendance ({attendance}%) shows consistent performance.")
        else:
            reasons.append(f"⚠ Attendance ({attendance}%) is below expected 75% benchmark.")

        if internal_marks >= 12:
            reasons.append(f"✔ Strong internal marks ({internal_marks}/20).")
        else:
            reasons.append(f"⚠ Internal marks ({internal_marks}/20) are lower than expected.")

        if backlogs == 0:
            reasons.append("✔ No backlogs — positive academic progress.")
        elif backlogs <= 2:
            reasons.append(f"⚠ Some backlogs ({backlogs}). Try clearing them soon.")
        else:
            reasons.append(f"❌ Too many backlogs ({backlogs}) affecting chances.")

        if extra_curriculars == 1:
            reasons.append("✔ Participation in extra-curricular activities helps skill growth.")
        else:
            reasons.append("ℹ Try joining extra-curriculars to improve your profile.")

        if part_time_job == 1:
            reasons.append("✔ Work experience adds practical and soft skills.")

        # ✅ Final result message
        if prediction == 1:
            result_text = f"✅ Eligible for Placement ({probability:.2f}%)"
        else:
            result_text = f"❌ Not Eligible for Placement ({probability:.2f}%)"

        return render_template(
            "predict.html",
            prediction_text=result_text,
            explanation=reasons  # Pass full explanation list
        )

    except Exception as e:
        return str(e)


@app.route("/insights")
def insights():
    try:
        generate_graphs("StudentAcademicData.csv")  # ✅ Refresh graphs every time
    except Exception as e:
        print("Graph Generation Error:", e)

    return render_template("insights.html",
        graph1="graph_attendance.png",
        graph2="graph_performance.png",
        graph3="placement_rate.png",
        graph4="marks_age_heatmap.png",
        graph5="correlation_matrix.png"
    )


@app.route("/about")
def about():
    return render_template("about.html")


if __name__ == "__main__":
    app.run(debug=True)
