# seed_zone.py (한 번만 실행)
from src.database import SessionLocal
from src.model import Zone
from sqlalchemy import text

with SessionLocal() as db:
    db.execute(text("DELETE FROM zone"))
    db.add_all([
        Zone(zone_id=1, zone_name="성북구", zone_lat=37.589, zone_lng=127.018, demand_qty=0),
        Zone(zone_id=2, zone_name="종로구", zone_lat=37.573, zone_lng=126.979, demand_qty=0),
    ])
    db.commit()
print("✅ seeded zone")
