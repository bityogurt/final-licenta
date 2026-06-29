from db import db_instance, Furniture, create_tables

def seed_generic_products():
    print("[SEED] Se adauga produse generice in baza de date...")

    generic_products = [
        {
            "name": "Scaun birou ergonomic",
            "category": "Chairs",
            "width_cm": 60.0, "depth_cm": 55.0, "height_cm": 90.0,
            "material_structure": "Metal",
            "upholstery_material": "Textil",
            "is_extensible": False, "seat_count": 1, "table_capacity_persons": 0,
            "base_cost": 350.0, "stock": 8,
        },
        {
            "name": "Scaun lemn masiv stejar",
            "category": "Chairs",
            "width_cm": 45.0, "depth_cm": 45.0, "height_cm": 85.0,
            "material_structure": "Lemn Masiv",
            "upholstery_material": "Fara",
            "is_extensible": False, "seat_count": 1, "table_capacity_persons": 0,
            "base_cost": 220.0, "stock": 12,
        },
        {
            "name": "Scaun bar inalt metal",
            "category": "Chairs",
            "width_cm": 40.0, "depth_cm": 40.0, "height_cm": 105.0,
            "material_structure": "Metal",
            "upholstery_material": "Textil",
            "is_extensible": False, "seat_count": 1, "table_capacity_persons": 0,
            "base_cost": 180.0, "stock": 15,
        },
        {
            "name": "Canapea extensibila 3 locuri",
            "category": "Sofas & armchairs",
            "width_cm": 220.0, "depth_cm": 90.0, "height_cm": 85.0,
            "material_structure": "Lemn Masiv",
            "upholstery_material": "Textil",
            "is_extensible": True, "seat_count": 3, "table_capacity_persons": 0,
            "base_cost": 1800.0, "stock": 3,
        },
        {
            "name": "Fotoliu piele naturala",
            "category": "Sofas & armchairs",
            "width_cm": 90.0, "depth_cm": 85.0, "height_cm": 95.0,
            "material_structure": "Lemn Masiv",
            "upholstery_material": "Piele",
            "is_extensible": False, "seat_count": 1, "table_capacity_persons": 0,
            "base_cost": 1200.0, "stock": 5,
        },
        {
            "name": "Canapea 2 locuri metal textil",
            "category": "Sofas & armchairs",
            "width_cm": 150.0, "depth_cm": 80.0, "height_cm": 80.0,
            "material_structure": "Metal",
            "upholstery_material": "Textil",
            "is_extensible": False, "seat_count": 2, "table_capacity_persons": 0,
            "base_cost": 800.0, "stock": 7,
        },
        {
            "name": "Masa dining extensibila 6 locuri",
            "category": "Tables & desks",
            "width_cm": 160.0, "depth_cm": 90.0, "height_cm": 75.0,
            "material_structure": "MDF/PAL",
            "upholstery_material": "Fara",
            "is_extensible": True, "seat_count": 0, "table_capacity_persons": 6,
            "base_cost": 900.0, "stock": 4,
        },
        {
            "name": "Birou MDF alb",
            "category": "Tables & desks",
            "width_cm": 120.0, "depth_cm": 60.0, "height_cm": 75.0,
            "material_structure": "MDF/PAL",
            "upholstery_material": "Fara",
            "is_extensible": False, "seat_count": 0, "table_capacity_persons": 2,
            "base_cost": 450.0, "stock": 6,
        },
        {
            "name": "Masa lemn masiv rotunda",
            "category": "Tables & desks",
            "width_cm": 100.0, "depth_cm": 100.0, "height_cm": 75.0,
            "material_structure": "Lemn Masiv",
            "upholstery_material": "Fara",
            "is_extensible": False, "seat_count": 0, "table_capacity_persons": 4,
            "base_cost": 700.0, "stock": 5,
        },
    ]

    with db_instance.atomic():
        for p in generic_products:
            _, created = Furniture.get_or_create(
                name=p["name"],
                defaults={
                    "category": p["category"],
                    "width_cm": p["width_cm"],
                    "depth_cm": p["depth_cm"],
                    "height_cm": p["height_cm"],
                    "material_structure": p["material_structure"],
                    "upholstery_material": p["upholstery_material"],
                    "is_extensible": p["is_extensible"],
                    "seat_count": p["seat_count"],
                    "table_capacity_persons": p["table_capacity_persons"],
                    "base_cost": p["base_cost"],
                    "stock_quantity": p["stock"],
                }
            )
            if created:
                print(f"[+] Adaugat: {p['name']} (Stoc: {p['stock']} buc)")

    print("[SEED] Popularea cu produse generice s-a finalizat.")

if __name__ == "__main__":
    print("Rulez script-ul de populare produse...")
    create_tables()
    db_instance.connect(reuse_if_open=True)
    seed_generic_products()
    db_instance.close()
    print("Script finalizat.")
