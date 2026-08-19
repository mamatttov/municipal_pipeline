"""
Генератор "сырых" исходников для муниципального пайплайна.
Три источника, как будто из разных систем учёта города/района:
  - utility_payments   - коммунальные платежи по домохозяйствам
  - citizen_complaints - обращения жителей (жалобы, заявки)
  - budget_execution   - план/факт исполнения бюджета по статьям

Данные намеренно грязные и разнородные по формату - как из реальных
разных ведомственных систем, которые никогда не согласовывают схемы между собой.
"""

import pandas as pd
import numpy as np
from pathlib import Path

N_HOUSEHOLDS = 20_000
N_PAYMENTS = 240_000
N_COMPLAINTS = 15_000
SEED = 7

rng = np.random.default_rng(SEED)
OUTPUT_DIR = Path("data/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DISTRICTS = ["Ленинский", "Октябрьский", "Свердловский", "Первомайский", "Аламудунский"]
DISTRICT_WEIGHTS = [0.28, 0.22, 0.20, 0.18, 0.12]

SERVICE_TYPES = ["Отопление", "Водоснабжение", "Электричество", "Вывоз мусора", "Газ"]

COMPLAINT_CATEGORIES = [
    "Дороги и тротуары", "Освещение", "Мусор и санитария",
    "Водоснабжение", "Отопление", "Благоустройство дворов", "Прочее",
]

COMPLAINT_STATUSES = ["Новое", "В работе", "Решено", "Отклонено"]

BUDGET_CATEGORIES = [
    "Благоустройство", "Дорожное хозяйство", "ЖКХ", "Образование",
    "Здравоохранение", "Соцзащита", "Управление",
]


def random_dates(start, end, size):
    start_ts = pd.Timestamp(start).value // 10**9
    end_ts = pd.Timestamp(end).value // 10**9
    return pd.to_datetime(rng.integers(start_ts, end_ts, size), unit="s")


# ---------- households (справочник, не отдельный источник, но нужен для генерации) ----------

household_ids = np.arange(1, N_HOUSEHOLDS + 1)
household_district = rng.choice(DISTRICTS, N_HOUSEHOLDS, p=DISTRICT_WEIGHTS)
household_area = np.round(rng.normal(58, 20, N_HOUSEHOLDS).clip(18, 220), 1)


# ---------- 1. utility_payments ----------

print("Генерирую utility_payments...")

payment_ids = np.arange(1, N_PAYMENTS + 1)
payment_household = rng.choice(household_ids, N_PAYMENTS)
payment_service = rng.choice(SERVICE_TYPES, N_PAYMENTS, p=[0.30, 0.20, 0.25, 0.15, 0.10])
payment_period = random_dates("2024-01-01", "2026-08-01", N_PAYMENTS)

# начисление примерно зависит от площади квартиры
area_by_household = pd.Series(household_area, index=household_ids)
household_area_for_payment = area_by_household.reindex(payment_household).to_numpy()
charged = household_area_for_payment * rng.uniform(15, 45, N_PAYMENTS)
charged = np.round(charged, 2)

# не все платят полностью и вовремя
paid_ratio = rng.beta(6, 1.3, N_PAYMENTS)  # большинство платит близко к 100%, но есть хвост недоплат
paid = np.round(charged * paid_ratio, 2)

payment_date = payment_period + pd.to_timedelta(rng.integers(0, 45, N_PAYMENTS), unit="D")

utility_payments = pd.DataFrame({
    "payment_id": payment_ids,
    "household_id": payment_household,
    "district": pd.Series(household_district, index=household_ids).reindex(payment_household).to_numpy(),
    "service_type": payment_service,
    "billing_period": payment_period.to_period("M").astype(str),
    "charged_amount": charged,
    "paid_amount": paid,
    "payment_date": payment_date,
})

# пропуски в дате оплаты - как будто платёж завис в системе и не разнесён
unpaid_idx = rng.choice(N_PAYMENTS, size=int(N_PAYMENTS * 0.05), replace=False)
utility_payments.loc[unpaid_idx, "payment_date"] = pd.NaT
utility_payments.loc[unpaid_idx, "paid_amount"] = 0.0

# дубли начислений - типичная история при сверке нескольких биллинговых систем
dupe_pay = utility_payments.sample(n=400, random_state=11)
utility_payments = pd.concat([utility_payments, dupe_pay], ignore_index=True)

utility_payments.to_csv(OUTPUT_DIR / "utility_payments.csv", index=False, encoding="utf-8-sig")
print(f"utility_payments.csv: {len(utility_payments):,} строк")


# ---------- 2. citizen_complaints ----------

print("Генерирую citizen_complaints...")

complaint_ids = np.arange(1, N_COMPLAINTS + 1)
complaint_district = rng.choice(DISTRICTS, N_COMPLAINTS, p=DISTRICT_WEIGHTS)
complaint_category = rng.choice(COMPLAINT_CATEGORIES, N_COMPLAINTS)
created_date = random_dates("2024-01-01", "2026-08-01", N_COMPLAINTS)

status = rng.choice(COMPLAINT_STATUSES, N_COMPLAINTS, p=[0.08, 0.12, 0.65, 0.15])

# срок решения зависит от категории - дороги решаются дольше, освещение быстрее
resolution_days_base = {
    "Дороги и тротуары": 25, "Освещение": 6, "Мусор и санитария": 4,
    "Водоснабжение": 8, "Отопление": 10, "Благоустройство дворов": 20, "Прочее": 12,
}
base_days = np.array([resolution_days_base[c] for c in complaint_category])
resolution_days = rng.poisson(base_days).clip(1, 120)

resolved_date = created_date + pd.to_timedelta(resolution_days, unit="D")
resolved_date = pd.Series(resolved_date)
resolved_date[status != "Решено"] = pd.NaT

citizen_complaints = pd.DataFrame({
    "complaint_id": complaint_ids,
    "district": complaint_district,
    "category": complaint_category,
    "status": status,
    "created_date": created_date,
    "resolved_date": resolved_date,
})

# опечатки в районах - операторы кол-центра вводят вручную
district_typo_map = {"Ленинский": ["ленинский", "Ленинский "], "Аламудунский": ["Аламудинский", "аламудунский"]}
typo_idx = rng.choice(N_COMPLAINTS, size=int(N_COMPLAINTS * 0.04), replace=False)
for i in typo_idx:
    d = citizen_complaints.at[i, "district"]
    if d in district_typo_map:
        citizen_complaints.at[i, "district"] = rng.choice(district_typo_map[d])

citizen_complaints.to_csv(OUTPUT_DIR / "citizen_complaints.csv", index=False, encoding="utf-8-sig")
print(f"citizen_complaints.csv: {len(citizen_complaints):,} строк")


# ---------- 3. budget_execution ----------

print("Генерирую budget_execution...")

rows = []
for year in [2024, 2025, 2026]:
    for month in range(1, 13):
        if year == 2026 and month > 7:
            continue
        for category in BUDGET_CATEGORIES:
            planned = rng.uniform(2_000_000, 25_000_000)
            # исполнение обычно 70-105% от плана, иногда сильный недобор
            exec_ratio = rng.beta(5, 1.8)
            actual = planned * exec_ratio
            rows.append({
                "year": year,
                "month": month,
                "category": category,
                "planned_amount": round(planned, 2),
                "actual_amount": round(actual, 2),
            })

budget_execution = pd.DataFrame(rows)
budget_execution.to_csv(OUTPUT_DIR / "budget_execution.csv", index=False, encoding="utf-8-sig")
print(f"budget_execution.csv: {len(budget_execution):,} строк")

print("\nГотово. Исходники в", OUTPUT_DIR.absolute())
