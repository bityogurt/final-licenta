# ml_models/ - fisiere pentru integrarea in backend Flask

Pune acest folder la radacina proiectului backend (langa `app.py` si `db.py`).

## Continut

| Fisier | Rol |
|---|---|
| `rf_optimizat.pkl` | Modelul Random Forest optimizat (GridSearchCV). R2=0.896, MAE=307 RON pe setul de test. |
| `scaler_ikea.pkl` | StandardScaler fitted pe cele 12 coloane de input, in ordinea din `feature_cols.pkl`. |
| `le_category.pkl` | LabelEncoder pentru `category` (Chairs=0, Sofas & armchairs=1, Tables & desks=2). |
| `le_material_structure.pkl` | LabelEncoder pentru `material_structure` (Lemn Masiv=0, MDF/PAL=1, Metal=2, Necunoscut=3). |
| `le_upholstery_material.pkl` | LabelEncoder pentru `upholstery_material` (Fara=0, Necunoscut=1, Piele=2, Textil=3). |
| `feature_cols.pkl` | Lista ordonata a celor 12 coloane de input, EXACT in ordinea ceruta de scaler/model. |

## Ordinea exacta a coloanelor (ordinea nu trebuie schimbata)

```python
['width_cm', 'depth_cm', 'height_cm', 'amprenta_sol_cm2', 'volum_teoretic_cm3',
 'proportie_w_h', 'category_enc', 'material_structure_enc',
 'upholstery_material_enc', 'is_extensible', 'seat_count', 'table_capacity_persons']
```

## Exemplu minimal de utilizare

```python
import joblib
import pandas as pd

rf = joblib.load('ml_models/rf_optimizat.pkl')
scaler = joblib.load('ml_models/scaler_ikea.pkl')
le_category = joblib.load('ml_models/le_category.pkl')
le_material = joblib.load('ml_models/le_material_structure.pkl')
le_upholstery = joblib.load('ml_models/le_upholstery_material.pkl')
feature_cols = joblib.load('ml_models/feature_cols.pkl')

def predict_price(width_cm, depth_cm, height_cm, category,
                   material_structure, upholstery_material,
                   is_extensible, seat_count, table_capacity_persons):
    amprenta_sol_cm2 = width_cm * depth_cm
    volum_teoretic_cm3 = width_cm * depth_cm * height_cm
    proportie_w_h = width_cm / (height_cm + 1e-5)

    row = {
        'width_cm': width_cm, 'depth_cm': depth_cm, 'height_cm': height_cm,
        'amprenta_sol_cm2': amprenta_sol_cm2,
        'volum_teoretic_cm3': volum_teoretic_cm3,
        'proportie_w_h': proportie_w_h,
        'category_enc': le_category.transform([category])[0],
        'material_structure_enc': le_material.transform([material_structure])[0],
        'upholstery_material_enc': le_upholstery.transform([upholstery_material])[0],
        'is_extensible': int(is_extensible),
        'seat_count': seat_count,
        'table_capacity_persons': table_capacity_persons,
    }
    # foloseste DataFrame cu feature_cols ca sa eviti warning-ul "X does not have valid feature names"
    X_df = pd.DataFrame([row])[feature_cols]
    X_scaled = scaler.transform(X_df)
    return float(rf.predict(X_scaled)[0])

# Exemplu testat:
preview = predict_price(
    width_cm=60.0, depth_cm=55.0, height_cm=90.0,
    category="Chairs", material_structure="Lemn Masiv",
    upholstery_material="Textil", is_extensible=0,
    seat_count=1, table_capacity_persons=0,
)
print(preview)  # -> 332.54 (RON)
```

## Valori valide pentru campurile categorice (validare la nivel de API)

- `category`: `"Chairs"`, `"Sofas & armchairs"`, `"Tables & desks"`
- `material_structure`: `"Lemn Masiv"`, `"MDF/PAL"`, `"Metal"`, `"Necunoscut"`
- `upholstery_material`: `"Fara"`, `"Piele"`, `"Textil"`, `"Necunoscut"`

Orice alta valoare va da eroare la `.transform()` (ValueError: y contains
previously unseen labels) -- valideaza inputul in endpoint INAINTE de a
apela encoderele, si returneaza HTTP 400 cu mesaj clar daca valoarea nu e
in lista de mai sus.
