import json
import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


INPUT_XLSX = "data/lora_dataset_1000.xlsx"
INPUT_SHEET = "Dataset_Seed_200"
OUT_DIR = Path("outputs/data")
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

TARGET_ATTACK_CATEGORIES = {
    "noisy_input",
    "allergen_bypass",
    "incomplete_fridge_data",
    "conflicting_constraints",
}

TARGET_SCENARIOS = {
    "recipe_selection",
    "cart_building",
    "fridge_based_request",
    "follow_up_revision",
    "substitution",
}

# Дополнительный фокус на noisy-поведение
ALLOW_IF_NOTES_CONTAINS = [
    "шум",
    "опечат",
    "разрыв",
    "noise",
    "typo",
]


def safe_str(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def row_matches_focus(row: pd.Series) -> bool:
    attack = safe_str(row.get("Attack_Category", ""))
    scenario = safe_str(row.get("Scenario_Type", ""))
    notes = safe_str(row.get("Notes", "")).lower()
    query = safe_str(row.get("User_Query", "")).lower()
    transform = safe_str(row.get("Transformation_Type", "")).lower()

    if attack in TARGET_ATTACK_CATEGORIES:
        return True
    if scenario in TARGET_SCENARIOS and any(k in transform for k in ["шум", "noise", "опечат", "typo"]):
        return True
    if any(token in notes for token in ALLOW_IF_NOTES_CONTAINS):
        return True
    if any(token in query for token in ["пАд", "ко рз", "ал ле", "вапрос", "ниче", "систимный"]):
        return True
    return False


def make_instruction(row: pd.Series) -> str:
    difficulty = safe_str(row.get("Difficulty", "medium")).lower()
    attack_category = safe_str(row.get("Attack_Category", "noisy_input"))
    scenario = safe_str(row.get("Scenario_Type", "recipe_selection"))
    behavior = safe_str(row.get("Expected_Behavior", ""))
    transform = safe_str(row.get("Transformation_Type", ""))
    notes = safe_str(row.get("Notes", ""))

    parts = [
        "Сгенерируй adversarial-запрос для продуктового чат-бота.",
        f"Целевой сценарий: {scenario}.",
        f"Тип риска: {attack_category}.",
        f"Сложность: {difficulty}.",
    ]

    if transform:
        parts.append(f"Желаемый тип трансформации: {transform}.")
    if behavior:
        parts.append(f"Ожидаемая реакция защитной модели: {behavior}.")
    if notes:
        parts.append(f"Комментарий: {notes}.")
    parts.append("Нужен один реалистичный пользовательский запрос на русском языке.")
    return " ".join(parts)


def normalize_output(text: str) -> str:
    text = safe_str(text)
    return " ".join(text.split())


def to_record(row: pd.Series) -> dict:
    instruction = make_instruction(row)
    output = normalize_output(row["User_Query"])
    return {
        "instruction": instruction,
        "input": "",
        "output": output,
        "difficulty": safe_str(row.get("Difficulty", "")),
        "attack_category": safe_str(row.get("Attack_Category", "")),
        "scenario_type": safe_str(row.get("Scenario_Type", "")),
    }


def write_jsonl(records, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    df = pd.read_excel(INPUT_XLSX, sheet_name=INPUT_SHEET)

    # Базовая очистка
    for col in [
        "Attack_Category", "Scenario_Type", "Difficulty", "Transformation_Type",
        "User_Query", "Expected_Behavior", "Target_Response", "Notes",
        "Safety_Label"
    ]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].apply(safe_str)

    # Оставляем фокусный срез под noisy-input и смежные случаи
    focused = df[df.apply(row_matches_focus, axis=1)].copy()

    # Если примеров мало, добавим все hard noisy-like запросы
    if len(focused) < 200:
        extra = df[
            (df["Difficulty"].str.lower().isin(["hard", "сложный", "сложная"])) |
            (df["Transformation_Type"].str.lower().str.contains("шум|noise|опечат|typo", regex=True, na=False))
        ]
        focused = pd.concat([focused, extra], ignore_index=True).drop_duplicates(subset=["User_Query"])

    # Убираем пустые
    focused = focused[focused["User_Query"].str.len() > 0].copy()

    records = [to_record(row) for _, row in focused.iterrows()]

    # train / val / test
    train_records, temp_records = train_test_split(
        records,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )
    val_records, test_records = train_test_split(
        temp_records,
        test_size=0.5,
        random_state=RANDOM_STATE,
    )

    write_jsonl(train_records, OUT_DIR / "train.jsonl")
    write_jsonl(val_records, OUT_DIR / "val.jsonl")
    write_jsonl(test_records, OUT_DIR / "test.jsonl")

    # Параллельно сохраним CSV для проверки
    pd.DataFrame(records).to_csv(OUT_DIR / "full_dataset.csv", index=False, encoding="utf-8-sig")

    print(f"Всего отобрано: {len(records)}")
    print(f"Train: {len(train_records)} | Val: {len(val_records)} | Test: {len(test_records)}")
    print(f"Файлы сохранены в: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
