from peewee import *
import os
import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, 'data.db')

db_instance = SqliteDatabase(DB_PATH)

print(f"Baza de date este activa la: {DB_PATH}")

class BaseModel(Model):
    class Meta:
        database = db_instance

class Furniture(BaseModel):
    name = CharField(unique=True)
    category = CharField()              # "Chairs" | "Sofas & armchairs" | "Tables & desks"
    width_cm = FloatField()
    depth_cm = FloatField()
    height_cm = FloatField()
    material_structure = CharField()    # "Lemn Masiv" | "MDF/PAL" | "Metal" | "Necunoscut"
    upholstery_material = CharField()   # "Fara" | "Piele" | "Textil" | "Necunoscut"
    is_extensible = BooleanField(default=False)
    seat_count = IntegerField(default=0)
    table_capacity_persons = IntegerField(default=0)
    stock_quantity = IntegerField(default=0)
    base_cost = FloatField(null=True)
    suggested_price = FloatField(null=True)  # rezultatul modelului ML
    selling_price = FloatField(null=True)    # pretul de vanzare real (folosit pentru re-antrenare)

    @property
    def amprenta_sol_cm2(self):
        return self.width_cm * self.depth_cm

    @property
    def volum_teoretic_cm3(self):
        return self.width_cm * self.depth_cm * self.height_cm

    @property
    def proportie_w_h(self):
        return self.width_cm / (self.height_cm + 1e-5)

class Material(BaseModel):
    name = CharField(unique=True)
    stock_quantity = FloatField()
    unit = CharField()
    cost_per_unit = FloatField()

class BOM(BaseModel):
    furniture = ForeignKeyField(Furniture, backref='materials_needed')
    material = ForeignKeyField(Material, backref='used_in_products')
    quantity_needed = FloatField()

class Order(BaseModel):
    customer_name = CharField()
    furniture = ForeignKeyField(Furniture, backref='orders')
    order_quantity = IntegerField()
    order_date = DateTimeField(default=datetime.datetime.now)
    status = CharField(default="Pending")

class PriceHistory(BaseModel):
    furniture = ForeignKeyField(Furniture, backref='price_history')
    market_price = FloatField()
    date_scraped = DateTimeField(constraints=[SQL('DEFAULT CURRENT_TIMESTAMP')])

class CompetitorLink(BaseModel):
    furniture = ForeignKeyField(Furniture, backref='competitor_links')
    store_name = CharField()
    url = CharField()

def create_tables():
    from resources.user import User
    with db_instance:
        db_instance.create_tables([
            User,
            Furniture,
            Material,
            BOM,
            Order,
            PriceHistory,
            CompetitorLink
        ], safe=True)

if __name__ == "__main__":
    try:
        create_tables()
        print(f"Succes! Baza de date '{DB_PATH}' a fost actualizata cu toate tabelele.")
    except Exception as e:
        print(f"Eroare la crearea tabelelor: {e}")
