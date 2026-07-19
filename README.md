"""
AnemiaAI (Multiclass) — Flask web app
Serves a CBC-based anemia TYPE classifier (9 diagnostic classes)
trained on dd.csv with Random Forest (see train.py).
"""
import os
import joblib
import numpy as np
from flask import Flask, render_template, request, jsonify

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

app = Flask(__name__)

# ------------------------------------------------------------------
# Load model artifacts once at startup
# ------------------------------------------------------------------
model = joblib.load(os.path.join(MODEL_DIR, "best_anemia_model.pkl"))
label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
feature_names = joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl"))
feature_medians = joblib.load(os.path.join(MODEL_DIR, "feature_medians.pkl"))
best_model_name = joblib.load(os.path.join(MODEL_DIR, "best_model_name.pkl"))

# Only Logistic Regression needs the scaler; Random Forest / XGBoost use raw values.
scaler = None
if best_model_name == "Logistic Regression":
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))

# ------------------------------------------------------------------
# Field metadata: unit + normal adult reference range + Arabic label
# (shown in the UI; matches the structure of the original AnemiaAI form)
# ------------------------------------------------------------------
FIELD_INFO = {
    "WBC":   {"label": "كريات الدم البيضاء",        "unit": "10³/μL", "range": (4, 11)},
    "RBC":   {"label": "كريات الدم الحمراء",          "unit": "10⁶/μL", "range": (3.8, 5.8)},
    "HGB":   {"label": "الهيموجلوبين",                "unit": "g/dL",   "range": (12, 17.5)},
    "HCT":   {"label": "الهيماتوكريت",                "unit": "%",      "range": (36, 52)},
    "MCV":   {"label": "متوسط حجم الكرة (MCV)",       "unit": "fL",     "range": (80, 100)},
    "MCH":   {"label": "متوسط هيموجلوبين الكرة (MCH)", "unit": "pg",     "range": (27, 34)},
    "MCHC":  {"label": "تركيز هيموجلوبين الكرة (MCHC)", "unit": "g/dL",   "range": (32, 36)},
    "PLT":   {"label": "الصفائح الدموية",             "unit": "10³/μL", "range": (150, 400)},
    "PDW":   {"label": "توزيع حجم الصفائح (PDW)",     "unit": "%",      "range": (9, 17)},
    "PCT":   {"label": "بلازما الصفائح (PCT)",        "unit": "%",      "range": (0.15, 0.4)},
    "LYMp":  {"label": "نسبة اللمفاويات (LYM%)",       "unit": "%",      "range": (20, 40)},
    "NEUTp": {"label": "نسبة النيوتروفيل (NEUT%)",     "unit": "%",      "range": (50, 70)},
    "LYMn":  {"label": "عدد اللمفاويات (LYM#)",        "unit": "10³/μL", "range": (1, 4)},
    "NEUTn": {"label": "عدد النيوتروفيل (NEUT#)",      "unit": "10³/μL", "range": (2, 7)},
}

# Short Arabic explanation shown with the result for each diagnosis class
DIAGNOSIS_INFO = {
    "Healthy": "لا توجد مؤشرات على فقر الدم بناءً على القيم المدخلة.",
    "Iron deficiency anemia": "فقر دم بسبب نقص الحديد — غالباً يرتبط بانخفاض MCV و MCH.",
    "Leukemia": "نمط القيم يتوافق مع حالات سرطان الدم؛ يتطلب مراجعة طبية فورية وتحاليل تأكيدية.",
    "Leukemia with thrombocytopenia": "نمط يتوافق مع سرطان الدم مصحوباً بانخفاض شديد في الصفائح الدموية؛ يتطلب مراجعة طبية فورية.",
    "Macrocytic anemia": "فقر دم بكريات كبيرة الحجم (MCV مرتفع) — قد يرتبط بنقص فيتامين B12 أو حمض الفوليك.",
    "Normocytic hypochromic anemia": "فقر دم بكريات طبيعية الحجم لكن منخفضة الصبغة (تركيز الهيموجلوبين منخفض).",
    "Normocytic normochromic anemia": "فقر دم بكريات طبيعية الحجم والصبغة — غالباً مرتبط بأمراض مزمنة أو نزيف حديث.",
    "Other microcytic anemia": "فقر دم بكريات صغيرة الحجم لا يعود بشكل نمطي لنقص الحديد فقط.",
    "Thrombocytopenia": "انخفاض في عدد الصفائح الدموية دون فقر دم واضح في الكريات الحمراء.",
}


def build_feature_vector(payload: dict) -> np.ndarray:
    """Fill missing fields with training-set medians, in the exact
    column order the model was trained on."""
    values = []
    for feat in feature_names:
        raw = payload.get(feat, "")
        if raw is None or str(raw).strip() == "":
            values.append(feature_medians[feat])
        else:
            values.append(float(raw))
    return np.array(values, dtype=float).reshape(1, -1)


@app.route("/")
def index():
    return render_template("index.html", fields=FIELD_INFO, feature_order=feature_names)


@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json(force=True) or {}

        all_empty = all(str(payload.get(feat, "")).strip() == "" for feat in feature_names)
        if all_empty:
            return jsonify({
                "ok": False,
                "error": "جميع الحقول فارغة. الرجاء إدخال قيمة واحدة على الأقل قبل التحليل.",
            }), 400

        x = build_feature_vector(payload)

        x_input = scaler.transform(x) if scaler is not None else x

        pred_idx = model.predict(x_input)[0]
        probs = model.predict_proba(x_input)[0]

        diagnosis = label_encoder.inverse_transform([pred_idx])[0]

        ranked = sorted(
            zip(label_encoder.classes_, probs),
            key=lambda t: t[1],
            reverse=True,
        )

        return jsonify({
            "ok": True,
            "diagnosis": diagnosis,
            "diagnosis_ar_note": DIAGNOSIS_INFO.get(diagnosis, ""),
            "confidence": round(float(probs[pred_idx]) * 100, 1),
            "ranking": [
                {"label": lbl, "probability": round(float(p) * 100, 1)}
                for lbl, p in ranked[:5]
            ],
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
