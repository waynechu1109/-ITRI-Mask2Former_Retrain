import json

# 載入 Mapillary 官方標籤（config_v2.0.json）
with open("mapillary_vistas/config_v2.0.json", "r", encoding="utf-8") as f:
    config = json.load(f)
official_class_names = set(label["name"] for label in config["labels"])

# 載入 ade2mapillary 對應表（你自己建立的）
with open("ade2mapillary_name_all.json", "r", encoding="utf-8") as f:
    ade2mapillary = json.load(f)
mapped_class_names = set(ade2mapillary.values())

# 找出所有你要的類別名稱
final_class_names = sorted(official_class_names.union(mapped_class_names))

# 建立 name → ID 對應（從 0 開始）
mapillary_name2id = {name: idx for idx, name in enumerate(final_class_names)}

# 寫入 JSON 檔案
with open("merged_dataset/mapillary_name2id.json", "w", encoding="utf-8") as f:
    json.dump(mapillary_name2id, f, indent=2, ensure_ascii=False)

print(f"已成功產生 mapillary_name2id.json，總類別數: {len(mapillary_name2id)}")
