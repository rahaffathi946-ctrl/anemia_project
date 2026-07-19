# AnemiaAI — تصنيف أنواع فقر الدم (Multiclass)

مشروع كامل: تدريب نموذج تعلّم آلي على بيانات CBC (`dd.csv`) لتصنيف تسعة
أنواع تشخيصية، مع تطبيق Flask يقدّم نفس تجربة موقعك
(anemiaai.onrender.com) لكن بواجهة مُحدَّثة للنموذج متعدد الأصناف.

## نتائج التدريب (train.py)

| الموديل | Accuracy | F1 (Macro) | ROC-AUC (OvR) |
|---|---|---|---|
| Logistic Regression | 0.830 | 0.632 | 0.958 |
| **Random Forest (الأفضل ✅)** | **0.980** | **0.909** | **0.9997** |
| XGBoost | 0.980 | 0.895 | 0.9995 |

تم اختيار **Random Forest** تلقائياً لأنه حقق أعلى F1-score (macro).
أهم 5 مؤشرات تأثيراً: `MCV`, `HGB`, `MCH`, `WBC`, `PLT`.

## بنية المشروع

```
anemia_project/
├── app.py                  # خادم Flask (يحمّل النموذج ويعرض الواجهة)
├── train.py                # سكربت التدريب الكامل (نسخة غير-Colab من الكود الأصلي)
├── dd.csv                  # البيانات
├── model/                  # ملفات النموذج المحفوظة (.pkl)
│   ├── best_anemia_model.pkl
│   ├── scaler.pkl
│   ├── label_encoder.pkl
│   ├── feature_names.pkl
│   └── feature_medians.pkl # لتعويض الحقول الفارغة عند التنبؤ
├── templates/index.html    # واجهة الموقع (عربي RTL)
└── static/{css,js}         # التصميم والتفاعل
```

## التشغيل محلياً

```bash
pip install flask joblib scikit-learn xgboost imbalanced-learn numpy pandas
python app.py
# افتح المتصفح على http://127.0.0.1:5000
```

## إعادة تدريب النموذج (لو غيّرتِ البيانات)

```bash
cd anemia_project   # نفس المجلد الذي فيه dd.csv
python train.py
mv *.pkl model/
```

## ملاحظات مهمة

- **كل الحقول اختيارية** في نموذج الفحص: أي قيمة تُترك فارغة يتم تعويضها
  تلقائياً بالوسيط (median) المحسوب من بيانات التدريب — بنفس فلسفة موقعك الأصلي.
- النموذج الحالي **لا يستخدم الجنس أو العمر** لأن `dd.csv` لا يحتوي على هذين
  العمودين (بخلاف نسختك السابقة للتصنيف الثنائي).
- عدد سجلات مكررة تم حذفها: 49 من أصل 1281 صف.
- تم استبدال القيم غير المنطقية طبياً (مثل HGB أو MCV خارج المدى الطبيعي
  الممكن) بـ NaN ثم تعويضها بالوسيط — تماماً كما في الكود الذي أرسلتِه.
- هذه الأداة **تثقيفية فقط** وليست بديلاً عن تشخيص طبي.

## النشر على Render

نفس خطوات نشر مشروعك السابق:
1. ارفعي المجلد لمستودع GitHub.
2. أنشئي Web Service جديد على Render واربطيه بالمستودع.
3. Build Command: `pip install -r requirements.txt`
4. Start Command: `gunicorn app:app`
5. تأكدي من رفع مجلد `model/` كاملاً (ملفات .pkl) لأن Render لا يشغّل train.py تلقائياً.

(أنشئي ملف requirements.txt يحتوي: flask, joblib, scikit-learn, xgboost,
imbalanced-learn, numpy, pandas, gunicorn)
