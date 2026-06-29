import os
import atexit
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from peewee import DoesNotExist
from playhouse.shortcuts import model_to_dict
from werkzeug.security import generate_password_hash, check_password_hash
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

import db
from resources.user import User
from db import Furniture, Material, Order, PriceHistory, CompetitorLink, db_instance
from seed_generic_products import seed_generic_products
from scraper_service import scrape_product_url as _scrape_url

app = Flask(__name__)
CORS(app)

ML_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ml_models')

ml = {}
try:
    ml['rf'] = joblib.load(os.path.join(ML_DIR, 'rf_optimizat.pkl'))
    ml['scaler'] = joblib.load(os.path.join(ML_DIR, 'scaler_ikea.pkl'))
    ml['le_category'] = joblib.load(os.path.join(ML_DIR, 'le_category.pkl'))
    ml['le_material'] = joblib.load(os.path.join(ML_DIR, 'le_material_structure.pkl'))
    ml['le_upholstery'] = joblib.load(os.path.join(ML_DIR, 'le_upholstery_material.pkl'))
    ml['feature_cols'] = joblib.load(os.path.join(ML_DIR, 'feature_cols.pkl'))
    print("[ML] Modele incarcate cu succes.")
except Exception as e:
    print(f"[ML] Atentie: nu s-au putut incarca modelele: {e}")

VALID_CATEGORIES = {"Chairs", "Sofas & armchairs", "Tables & desks"}
VALID_MATERIALS = {"Lemn Masiv", "MDF/PAL", "Metal", "Necunoscut"}
VALID_UPHOLSTERY = {"Fara", "Piele", "Textil", "Necunoscut"}


def safe_encode(le, value):
    if value in le.classes_:
        return le.transform([value])[0]
    return le.transform([le.classes_[0]])[0]


def build_ml_row(p_or_data, is_dict=False):
    get = (lambda k: p_or_data.get(k)) if is_dict else (lambda k: getattr(p_or_data, k))
    w = float(get('width_cm') or 0)
    d = float(get('depth_cm') or 0)
    h = float(get('height_cm') or 0)
    return {
        'width_cm': w, 'depth_cm': d, 'height_cm': h,
        'amprenta_sol_cm2': w * d,
        'volum_teoretic_cm3': w * d * h,
        'proportie_w_h': w / (h + 1e-5),
        'category_enc': safe_encode(ml['le_category'], get('category')),
        'material_structure_enc': safe_encode(ml['le_material'], get('material_structure')),
        'upholstery_material_enc': safe_encode(ml['le_upholstery'], get('upholstery_material')),
        'is_extensible': int(bool(get('is_extensible'))),
        'seat_count': int(get('seat_count') or 0),
        'table_capacity_persons': int(get('table_capacity_persons') or 0),
    }




with db_instance:
    all_tables = [User, Furniture, Material, db.BOM, Order, PriceHistory, CompetitorLink]
    db_instance.create_tables(all_tables, safe=True)

    # Migrare: adaugam selling_price daca nu exista deja
    try:
        db_instance.execute_sql('ALTER TABLE furniture ADD COLUMN selling_price REAL')
        print("[DB] Coloana selling_price adaugata.")
    except Exception:
        pass  # coloana deja exista

    if Furniture.select().count() == 0:
        print("[DEV] Populez baza de date cu date de test...")
        seed_generic_products()

User.get_or_create(
    username="administrator",
    defaults={'password': generate_password_hash("admin123")}
)


# AUTENTIFICARE

@app.post("/login")
def login():
    data = request.json
    try:
        user = User.get(User.username == data.get("username"))
        if check_password_hash(user.password, data.get("password", "")):
            return jsonify({"status": "success", "message": "Login successful!"}), 200
        return jsonify({"status": "error", "message": "Invalid password"}), 401
    except DoesNotExist:
        return jsonify({"status": "error", "message": "User not found"}), 404

@app.post("/register")
def register():
    data = request.json
    reg_user = data.get("username")
    reg_password = data.get("password")
    try:
        User.get(User.username == reg_user)
        return jsonify({"status": "error", "message": "Username taken!"}), 400
    except DoesNotExist:
        if reg_password and len(reg_password) >= 4:
            User.create(username=reg_user, password=generate_password_hash(reg_password))
            return jsonify({"status": "success", "message": "User successfully registered!"}), 201
        return jsonify({"status": "error", "message": "Password too short"}), 400


# INVENTAR SI PRODUSE

@app.get("/api/inventory")
def get_inventory():
    products = [model_to_dict(p) for p in Furniture.select()]
    return jsonify({"products": products, "materials": []}), 200


@app.get("/products")
def get_products_for_ui():
    product_list = [
        {"id": p.id, "name": p.name, "base_cost": p.base_cost,
         "category": p.category, "stock_quantity": p.stock_quantity}
        for p in Furniture.select()
    ]
    return jsonify(product_list), 200


@app.post("/api/furniture")
def add_furniture():
    data = request.json

    required = ['name', 'category', 'width_cm', 'depth_cm', 'height_cm',
                'material_structure', 'upholstery_material']
    for field in required:
        if not data.get(field):
            return jsonify({"status": "error", "message": f"Lipseste campul: {field}"}), 400

    if data['category'] not in VALID_CATEGORIES:
        return jsonify({"status": "error", "message": f"Categorie invalida. Valori acceptate: {list(VALID_CATEGORIES)}"}), 400
    if data['material_structure'] not in VALID_MATERIALS:
        return jsonify({"status": "error", "message": f"Material invalid. Valori acceptate: {list(VALID_MATERIALS)}"}), 400
    if data['upholstery_material'] not in VALID_UPHOLSTERY:
        return jsonify({"status": "error", "message": f"Tapiterie invalida. Valori acceptate: {list(VALID_UPHOLSTERY)}"}), 400

    try:
        with db_instance.atomic():
            new_item = Furniture.create(
                name=data['name'],
                category=data['category'],
                width_cm=float(data['width_cm']),
                depth_cm=float(data['depth_cm']),
                height_cm=float(data['height_cm']),
                material_structure=data['material_structure'],
                upholstery_material=data['upholstery_material'],
                is_extensible=bool(data.get('is_extensible', False)),
                seat_count=int(data.get('seat_count', 0)),
                table_capacity_persons=int(data.get('table_capacity_persons', 0)),
                stock_quantity=int(data.get('stock_quantity', 0)),
                base_cost=float(data['base_cost']) if data.get('base_cost') else None,
                selling_price=float(data['selling_price']) if data.get('selling_price') else None,
            )

            # Calculeaza automat pretul ML si salveaza-l in suggested_price
            if ml:
                try:
                    X = pd.DataFrame([build_ml_row(new_item)])[ml['feature_cols']]
                    new_item.suggested_price = round(float(ml['rf'].predict(ml['scaler'].transform(X))[0]), 2)
                    new_item.save()
                except Exception:
                    pass

            return jsonify(model_to_dict(new_item)), 201
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


# PREDICTIE PRET ML

@app.post("/api/predict-price")
def predict_price():
    if not ml:
        return jsonify({"status": "error", "message": "Modelele ML nu sunt incarcate."}), 503

    data = request.json

    if data.get('category') not in VALID_CATEGORIES:
        return jsonify({"status": "error", "message": f"Categorie invalida. Valori acceptate: {list(VALID_CATEGORIES)}"}), 400
    if data.get('material_structure') not in VALID_MATERIALS:
        return jsonify({"status": "error", "message": f"Material invalid. Valori acceptate: {list(VALID_MATERIALS)}"}), 400
    if data.get('upholstery_material') not in VALID_UPHOLSTERY:
        return jsonify({"status": "error", "message": f"Tapiterie invalida. Valori acceptate: {list(VALID_UPHOLSTERY)}"}), 400

    try:
        X = pd.DataFrame([build_ml_row(data, is_dict=True)])[ml['feature_cols']]
        X_scaled = ml['scaler'].transform(X)
        predicted = float(ml['rf'].predict(X_scaled)[0])

        furniture_id = data.get('furniture_id')
        if furniture_id:
            try:
                item = Furniture.get_by_id(furniture_id)
                item.suggested_price = predicted
                item.save()
            except DoesNotExist:
                pass

        return jsonify({"predicted_price_ron": round(predicted, 2)}), 200

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


# ACTUALIZARE PRETURI ML PENTRU TOATE PRODUSELE

@app.post("/api/update-all-prices")
def update_all_prices():
    if not ml:
        return jsonify({"status": "error", "message": "Modelele ML nu sunt disponibile."}), 503

    updated = 0
    errors = 0
    for p in Furniture.select():
        try:
            row = build_ml_row(p)
            X = pd.DataFrame([row])[ml['feature_cols']]
            p.suggested_price = round(float(ml['rf'].predict(ml['scaler'].transform(X))[0]), 2)
            p.save()
            updated += 1
        except Exception as e:
            print(f"[ML] Eroare la produsul {p.id} ({p.name}): {e}")
            errors += 1

    return jsonify({"status": "success", "updated": updated, "errors": errors}), 200


# RE-ANTRENARE MODEL ML

@app.post("/api/retrain")
def retrain_model():
    ikea_csv = os.path.join(ML_DIR, 'ikea_training_data.csv')
    if not os.path.exists(ikea_csv):
        return jsonify({"status": "error", "message": "Fisierul de date IKEA lipseste din folderul ml_models."}), 404

    try:
        df_ikea = pd.read_csv(ikea_csv).rename(columns={'price_ron': 'target_price'})
        cols = ['category', 'width_cm', 'depth_cm', 'height_cm', 'material_structure',
                'upholstery_material', 'is_extensible', 'seat_count', 'table_capacity_persons', 'target_price']
        df_base = df_ikea[cols].copy()

        user_rows = [
            {
                'category': p.category,
                'width_cm': p.width_cm,
                'depth_cm': p.depth_cm,
                'height_cm': p.height_cm,
                'material_structure': p.material_structure,
                'upholstery_material': p.upholstery_material,
                'is_extensible': int(p.is_extensible),
                'seat_count': p.seat_count,
                'table_capacity_persons': p.table_capacity_persons,
                'target_price': p.selling_price,
            }
            for p in Furniture.select().where(Furniture.selling_price.is_null(False))
        ]
        n_ikea = len(df_base)
        n_user = len(user_rows)

        df_combined = pd.concat(
            [df_base, pd.DataFrame(user_rows)] if user_rows else [df_base],
            ignore_index=True
        )

        df_combined['amprenta_sol_cm2'] = df_combined['width_cm'] * df_combined['depth_cm']
        df_combined['volum_teoretic_cm3'] = df_combined['width_cm'] * df_combined['depth_cm'] * df_combined['height_cm']
        df_combined['proportie_w_h'] = df_combined['width_cm'] / (df_combined['height_cm'] + 1e-5)

        le_cat = LabelEncoder()
        le_mat = LabelEncoder()
        le_uph = LabelEncoder()
        df_combined['category_enc'] = le_cat.fit_transform(df_combined['category'])
        df_combined['material_structure_enc'] = le_mat.fit_transform(df_combined['material_structure'])
        df_combined['upholstery_material_enc'] = le_uph.fit_transform(df_combined['upholstery_material'])

        feature_cols = ['width_cm', 'depth_cm', 'height_cm', 'amprenta_sol_cm2',
                        'volum_teoretic_cm3', 'proportie_w_h', 'category_enc',
                        'material_structure_enc', 'upholstery_material_enc',
                        'is_extensible', 'seat_count', 'table_capacity_persons']

        X = df_combined[feature_cols]
        y = df_combined['target_price']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        rf = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
        rf.fit(X_train_s, y_train)

        r2 = round(float(r2_score(y_test, rf.predict(X_test_s))), 4)
        mae = round(float(mean_absolute_error(y_test, rf.predict(X_test_s))), 2)

        joblib.dump(rf, os.path.join(ML_DIR, 'rf_optimizat.pkl'))
        joblib.dump(scaler, os.path.join(ML_DIR, 'scaler_ikea.pkl'))
        joblib.dump(le_cat, os.path.join(ML_DIR, 'le_category.pkl'))
        joblib.dump(le_mat, os.path.join(ML_DIR, 'le_material_structure.pkl'))
        joblib.dump(le_uph, os.path.join(ML_DIR, 'le_upholstery_material.pkl'))
        joblib.dump(feature_cols, os.path.join(ML_DIR, 'feature_cols.pkl'))

        ml['rf'] = rf
        ml['scaler'] = scaler
        ml['le_category'] = le_cat
        ml['le_material'] = le_mat
        ml['le_upholstery'] = le_uph
        ml['feature_cols'] = feature_cols

        print(f"[ML] Model re-antrenat: R2={r2}, MAE={mae} RON, {n_ikea} IKEA + {n_user} user")
        return jsonify({
            "status":        "success",
            "r2_score":      r2,
            "mae_ron":       mae,
            "total_samples": len(df_combined),
            "ikea_samples":  n_ikea,
            "user_samples":  n_user,
        }), 200

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# STOC

@app.delete("/api/furniture/<int:furniture_id>")
def delete_furniture(furniture_id):
    try:
        item = Furniture.get_by_id(furniture_id)
        item.delete_instance()
        return jsonify({"status": "success"}), 200
    except DoesNotExist:
        return jsonify({"status": "error", "message": "Produs negasit"}), 404


@app.patch("/api/furniture/<int:furniture_id>/stock")
def update_stock(furniture_id):
    data = request.json
    new_stock = data.get('stock_quantity')
    if new_stock is None or int(new_stock) < 0:
        return jsonify({"status": "error", "message": "Stoc invalid"}), 400
    try:
        item = Furniture.get_by_id(furniture_id)
        item.stock_quantity = int(new_stock)
        item.save()
        return jsonify({"status": "success", "stock_quantity": item.stock_quantity}), 200
    except DoesNotExist:
        return jsonify({"status": "error", "message": "Produs negasit"}), 404


# COMENZI

@app.post("/api/orders")
def create_order():
    data = request.json
    furniture_id = data.get("furniture_id")
    quantity = int(data.get("quantity", 1))
    customer = data.get("customer_name", "Client Spontan")

    try:
        with db_instance.atomic():
            item = Furniture.get_by_id(furniture_id)
            if item.stock_quantity < quantity:
                return jsonify({"status": "error", "message": "Stoc insuficient pentru procesare"}), 400
            item.stock_quantity -= quantity
            item.save()
            new_order = Order.create(
                furniture=item,
                status="Finalizat",
                customer_name=customer,
                order_quantity=quantity
            )
            return jsonify({"status": "success", "order_id": new_order.id, "new_stock": item.stock_quantity}), 201
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# IMPORT PRODUS DIN URL COMPETITOR

@app.post("/api/scrape-product")
def scrape_product_endpoint():
    data = request.json or {}
    url = data.get('url', '').strip()
    if not url:
        return jsonify({"status": "error", "message": "URL necesar."}), 400
    result, msg = _scrape_url(url)
    return jsonify({"status": "ok", "data": result, "message": msg})


@app.post("/api/import-scraped-product")
def import_scraped_product():
    data = request.json or {}
    name = data.get('name', '').strip()
    if not name:
        return jsonify({"status": "error", "message": "Numele produsului este obligatoriu."}), 400
    if Furniture.select().where(Furniture.name == name).count():
        return jsonify({"status": "error", "message": f"Produsul '{name}' exista deja in baza de date."}), 409

    try:
        w = float(data.get('width_cm') or 0)
        d = float(data.get('depth_cm') or 0)
        h = float(data.get('height_cm') or 0)
        sp = float(data['selling_price']) if data.get('selling_price') else None

        with db_instance.atomic():
            item = Furniture.create(
                name=name,
                category=data.get('category', 'Chairs'),
                width_cm=w, depth_cm=d, height_cm=h,
                material_structure=data.get('material_structure', 'Necunoscut'),
                upholstery_material=data.get('upholstery_material', 'Fara'),
                is_extensible=False,
                seat_count=int(data.get('seat_count', 0)),
                table_capacity_persons=int(data.get('table_capacity_persons', 0)),
                stock_quantity=0,
                base_cost=None,
                selling_price=sp,
            )

        if ml:
            try:
                X = pd.DataFrame([build_ml_row(item)])[ml['feature_cols']]
                item.suggested_price = round(float(ml['rf'].predict(ml['scaler'].transform(X))[0]), 2)
                item.save()
            except Exception:
                pass

        return jsonify({"status": "success", "message": f"Produsul '{name}' a fost importat.", "id": item.id}), 201
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


def cleanup():
    if not db_instance.is_closed():
        db_instance.close()

atexit.register(cleanup)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
