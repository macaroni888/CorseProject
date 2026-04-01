import pandas as pd
pd.set_option("display.max_columns", None)

def get_loans_RK(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    header_i = next(i for i, s in enumerate(lines) if s.startswith(";Metric;"))

    df = pd.read_csv(
        path,
        sep=";",
        skiprows=header_i,
        encoding="utf-8",
    )

    if str(df.columns[0]).startswith("Unnamed"):
        df = df.drop(columns=df.columns[0])

    date_cols = [c for c in df.columns if c != "Metric"]
    values_num = df[date_cols].apply(pd.to_numeric, errors="coerce")

    mask_ts = values_num.notna().any(axis=1)

    raw_metrics = df.loc[mask_ts, "Metric"].astype(str).str.strip().tolist()
    clean_metrics = []
    context = ""

    for m in raw_metrics:
        m_l = m.lower()

        if "business" in m_l and "small" not in m_l:
            context = "biz"
        elif "individuals" in m_l:
            context = "ind"

        if m_l == "total":
            clean_metrics.append("loans_kz_total")
        elif m_l == "loans to business":
            clean_metrics.append("loans_kz_biz_total")
        elif m_l == "loans to individuals":
            clean_metrics.append("loans_kz_ind_total")
        elif m_l == "national currency":
            clean_metrics.append(f"loans_kz_{context}_kzt")
        elif m_l == "foreign currency":
            clean_metrics.append(f"loans_kz_{context}_foreign")
        elif m_l == "small business":
            clean_metrics.append("loans_kz_biz_small")
        elif m_l == "medium business":
            clean_metrics.append("loans_kz_biz_medium")
        elif m_l == "large business":
            clean_metrics.append("loans_kz_biz_large")
        elif m_l == "mortgage loans":
            clean_metrics.append("loans_kz_ind_mortgage")
        elif m_l == "consumer loans":
            clean_metrics.append("loans_kz_ind_consumer")
        elif m_l == "others":
            clean_metrics.append("loans_kz_ind_others")
        else:
            clean_metrics.append(f"skip_{m_l}")

    out = values_num.loc[mask_ts].copy()
    out.index = clean_metrics

    out = out[~out.index.str.startswith("skip_")]

    out = out.T
    out.index = pd.to_datetime(out.index, format="%Y-%m-%d", errors="raise")
    out = out.sort_index()

    out.index.name = "date"
    return out.reset_index()


def get_cpi_yoy_table(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    years_i = next(i for i, s in enumerate(lines) if s.lstrip().startswith(";2000;"))

    df = pd.read_csv(
        path,
        sep=";",
        skiprows=years_i,
        header=0,
        encoding="utf-8",
        decimal=",",
        engine="python",
    )

    first = df.columns[0]
    df = df.rename(columns={first: "category"})

    target_section = "декабрь к декабрю предыдущего года"

    s = (
        df["category"].astype(str)
        .str.replace("\u00A0", " ", regex=False)
        .str.strip()
    )
    df["section"] = s.where(s == target_section).ffill()

    df = df[df["section"] == target_section].copy()
    df = df[df["category"] != target_section].copy()

    year_cols = [c for c in df.columns if c not in ["section", "category"]]

    long = df.melt(
        id_vars=["category"],
        value_vars=year_cols,
        var_name="year",
        value_name="value",
    )

    long["year"] = pd.to_numeric(long["year"], errors="coerce")
    long = long[long["year"].notna()].copy()
    long["date"] = pd.to_datetime(
        long["year"].astype(int).astype(str) + "-01-01",
        format="%Y-%m-%d",
        errors="raise",
    )

    out = (
        long.pivot_table(
            index="date",
            columns="category",
            values="value",
            aggfunc="first",
        )
        .sort_index()
    )

    ru_to_en = {
        "Товары и услуги": "all",
        "Продовольственные товары": "food_total",
        "Непродовольственные товары": "non_food_total",
        "Платные услуги": "services_total",
        "Продукты питания": "food_only",
        "Безалкогольные напитки": "soft_drinks",
        "Алкогольные напитки и табачные изделия": "alcohol_tobacco",
        "Одежда и обувь": "clothing_footwear",
        "Жилищные услуги и другие виды топлива": "housing_utilities",
        "Предметы домашнего обихода": "household_items",
        "Здравоохранение": "health",
        "Транспорт": "transport",
        "Связь": "communication",
        "Отдых и культура": "recreation",
        "Образование": "education",
        "Рестораны и гостиницы": "hotels_restaurants",
        "Разные товары и услуги": "miscellaneous",
    }

    out.columns = [f"cpi_kz_eoy_{ru_to_en.get(c.strip(), c.strip().replace(' ', '_').lower())}" for c in out.columns]

    out = out.reset_index()
    out.columns.name = None

    return out


def get_zgb_cp(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    header_i = next(i for i, s in enumerate(lines) if s.lstrip().startswith("Дата;"))

    df = pd.read_csv(
        path,
        sep=";",
        skiprows=header_i,
        header=0,
        encoding="utf-8",
        decimal=",",
        engine="python",
    )

    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")].copy()
    df.columns = df.columns.astype(str).str.strip()

    if df.columns.has_duplicates:
        df.columns = pd.io.parsers.ParserBase({"names": df.columns})._maybe_dedup_names(df.columns)

    df["date"] = pd.to_datetime(
        df["Дата"].astype(str).str.strip(),
        format="%d.%m.%y",
        errors="raise",
    )

    rename_map = {
        "KZGB_CPs": "bond_kz_govt_cp_short_u1y",
        "KZGB_CPm": "bond_kz_govt_cp_mid_1-5y",
        "KZGB_CPl": "bond_kz_govt_cp_long_o5y",
        "KZGB_CP": "bond_kz_govt_cp_total"
    }

    df = df.rename(columns=rename_map)
    value_cols = list(rename_map.values())

    out = df[["date", *value_cols]].copy()
    out[value_cols] = out[value_cols].apply(pd.to_numeric, errors="coerce")

    out = out.sort_values("date").reset_index(drop=True)
    return out


def get_kase_bmc(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    header_i = next(i for i, s in enumerate(lines) if s.lstrip().startswith("Дата;"))

    df = pd.read_csv(
        path,
        sep=";",
        skiprows=header_i,
        header=0,
        encoding="utf-8",
        decimal=",",
        engine="python",
    )
    df = df.loc[:, ~df.columns.astype(str).str.match(r"^Unnamed", na=False)].copy()  # [web:423]

    df.columns = df.columns.astype(str).str.strip()
    df = df.rename(columns={"Дата": "date"})

    df["date"] = pd.to_datetime(
        df["date"].astype(str).str.strip(),
        format="%d.%m.%y",
        errors="raise",
    )

    value_cols = [c for c in df.columns if c != "date"]
    df[value_cols] = df[value_cols].apply(pd.to_numeric, errors="coerce")

    out = df[["date", *value_cols]].sort_values("date").reset_index(drop=True)
    return out


def get_gdp_per_capita_kz(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    header_i = next(i for i, s in enumerate(lines) if s.lstrip().startswith(";1990 год;"))

    df = pd.read_csv(
        path,
        sep=";",
        skiprows=header_i,
        header=0,
        encoding="utf-8",
        decimal=",",
        engine="python",
    )

    first = df.columns[0]
    df = df.rename(columns={first: "metric"})
    df["metric"] = df["metric"].astype(str).str.replace("\u00A0", " ", regex=False).str.strip()

    df = df[df["metric"].notna() & (df["metric"] != "")].copy()
    df = df[~df["metric"].str.startswith('"*', na=False)].copy()

    year_cols = [c for c in df.columns if c != "metric"]

    long = df.melt(
        id_vars=["metric"],
        value_vars=year_cols,
        var_name="year",
        value_name="value",
    )

    long["year"] = long["year"].astype(str).str.extract(r"(\d{4})", expand=False)
    long["year"] = pd.to_numeric(long["year"], errors="coerce")
    long = long[long["year"].notna()].copy()

    long["date"] = pd.to_datetime(
        long["year"].astype(int).astype(str) + "-01-01",
        format="%Y-%m-%d",
        errors="raise",
    )

    long["value"] = pd.to_numeric(long["value"], errors="coerce")

    out = (
        long.pivot_table(index="date", columns="metric", values="value", aggfunc="first")
            .sort_index()
    )

    ru_to_en = {
        "Курс доллара, тенге за 1 доллар США": "macro_kz_usd_kzt_avg",
        "Валовой внутренний продукт на душу населения, тенге": "macro_kz_gdp_per_capita_kzt",
        "Валовой внутренний продукт на душу населения, доллары США": "macro_kz_gdp_per_capita_usd",
        "ИФО Валового внутреннего продукта на душу населения, в % к предыдущему году": "macro_kz_gdp_per_capita_real_growth_pct",
    }

    out = out.rename(columns=lambda c: ru_to_en.get(str(c).strip()))
    out = out[[c for c in out.columns if c is not None]]

    out.columns.name = None
    out = out.reset_index()

    return out


def get_base_rate_kz(path):
    import pandas as pd

    df = pd.read_csv(
        path,
        sep=";",
        header=0,
        encoding="utf-8",
        engine="python",
        decimal=",",
        usecols=[0, 1],
    )

    df.columns = ["date", "macro_kz_base_rate"]

    df["date"] = df["date"].astype(str).str.replace("*", "", regex=False).str.strip()
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")

    df["macro_kz_base_rate"] = pd.to_numeric(df["macro_kz_base_rate"], errors="coerce")

    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    return df


def get_key_rate(path):
    df = pd.read_csv(
        path,
        encoding="utf-8",
        sep=";",
        decimal=",",
    )

    df["date"] = pd.to_datetime("01." + df["date"].astype(str), format="%d.%m.%Y", errors="raise")

    rename_map = {
        "key_rate": "macro_ru_key_rate",
        "inflation": "macro_ru_inflation_yoy"
    }

    df = df.rename(columns=rename_map)

    df_res = df[["date", "macro_ru_key_rate", "macro_ru_inflation_yoy"]]
    return df_res


def get_price_idx(path):
    import pandas as pd

    df = pd.read_csv(
        path,
        sep=";",
        skiprows=7,
        header=0,
        skipfooter=4,
        engine="python",
        decimal=",",
        encoding="utf-8",
    )

    s = df.iloc[:, 0].astype(str).str.strip()

    year = s.where(s.str.fullmatch(r"\d{4}")).ffill()
    qmask = s.str.contains("квартал", na=False)
    d = df.loc[qmask].copy()
    d["year"] = year.loc[qmask].values

    roman_to_q = {"I": 1, "II": 2, "III": 3, "IV": 4}
    q_roman = d.iloc[:, 0].str.extract(r"^\s*(I{1,3}|IV)\s+квартал", expand=False)
    q_num = q_roman.map(roman_to_q)

    month = (q_num - 1) * 3 + 1
    d["date"] = pd.to_datetime(
        d["year"] + "-" + month.astype(int).astype(str).str.zfill(2) + "-01",
        format="%Y-%m-%d",
        errors="raise",
    )

    rename_map = {
        df.columns[3]: "macro_ru_cpi_all_yoy",
        df.columns[6]: "macro_ru_cpi_food_yoy",
        df.columns[9]: "macro_ru_cpi_food_excl_alc_yoy",
        df.columns[12]: "macro_ru_cpi_nonfood_yoy",
        df.columns[15]: "macro_ru_cpi_services_yoy"
    }

    d = d.rename(columns=rename_map)
    value_cols = list(rename_map.values())

    out = d.loc[:, ["date", *value_cols]].copy()

    for c in value_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out = out.sort_values("date").reset_index(drop=True)
    return out


def get_yield_curve(path):
    import pandas as pd

    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    header_i = next(i for i, s in enumerate(lines) if s.lstrip().lower().startswith("tradedate;"))

    df = pd.read_csv(
        path,
        sep=";",
        skiprows=header_i,
        header=0,
        decimal=",",
        encoding="utf-8",
    )

    df.columns = df.columns.astype(str).str.strip().str.lower()

    df["date"] = pd.to_datetime(
        df["tradedate"].astype(str).str.strip(),
        format="%d.%m.%Y",
        errors="raise",
    )

    rename_map = {
        "period_0.25": "bond_ru_ofz_yield_3m",
        "period_0.5": "bond_ru_ofz_yield_6m",
        "period_0.75": "bond_ru_ofz_yield_9m",
        "period_1.0": "bond_ru_ofz_yield_1y",
        "period_2.0": "bond_ru_ofz_yield_2y",
        "period_3.0": "bond_ru_ofz_yield_3y",
        "period_5.0": "bond_ru_ofz_yield_5y",
        "period_7.0": "bond_ru_ofz_yield_7y",
        "period_10.0": "bond_ru_ofz_yield_10y",
        "period_15.0": "bond_ru_ofz_yield_15y",
        "period_20.0": "bond_ru_ofz_yield_20y"
    }

    actual_map = {old: new for old, new in rename_map.items() if old in df.columns}
    df = df.rename(columns=actual_map)

    df = df.sort_values(["date", "tradetime"]).drop_duplicates(["date"], keep="last")

    value_cols = list(actual_map.values())
    out = df[["date", *value_cols]].copy()

    for c in value_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    return out.sort_values("date").reset_index(drop=True)


def get_labor_force(path):
    import pandas as pd
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    df = pd.read_csv(
        path,
        sep=";",
        skiprows=6,
        skipfooter=2,
        header=0,
        encoding="utf-8",
        decimal=",",
        engine="python",
    )

    df.columns = df.columns.astype(str).str.strip()
    first_col = df.columns[0]
    s = df[first_col].astype(str).str.strip()

    year = s.where(s.str.fullmatch(r"\d{4}")).ffill()

    month_map = {
        "январь": 1, "февраль": 2, "март": 3, "апрель": 4, "май": 5, "июнь": 6,
        "июль": 7, "август": 8, "сентябрь": 9, "октябрь": 10, "ноябрь": 11, "декабрь": 12,
    }

    month_name = (
        s.str.lower()
        .str.replace("\u00A0", " ", regex=False)
        .str.extract(r"^([а-яё]+)", expand=False)
    )
    month_num = month_name.map(month_map)

    mask_month = month_num.notna()
    d = df.loc[mask_month].copy()
    d["year"] = year.loc[mask_month].values
    d["month"] = month_num.loc[mask_month].astype(int)

    d["date"] = pd.to_datetime(
        d["year"].astype(str) + "-" + d["month"].astype(str).str.zfill(2) + "-01",
        format="%Y-%m-%d",
        errors="raise",
    )

    rename_map = {
        df.columns[1]: "macro_ru_labor_force",
        df.columns[2]: "macro_ru_labor_employed",
        df.columns[3]: "macro_ru_labor_unemployed",
        df.columns[4]: "macro_ru_labor_participation_rate",
        df.columns[5]: "macro_ru_labor_employment_rate",
        df.columns[6]: "macro_ru_labor_unemployment_rate"
    }

    d = d.rename(columns=rename_map)
    value_cols = list(rename_map.values())

    out = d[["date", *value_cols]].copy()

    for c in value_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    return out.sort_values("date").reset_index(drop=True)


def get_ruonia(path):
    df = pd.read_csv(
        path,
        sep=";",
        encoding="utf-8",
        dtype=str,
        skipinitialspace=True,
    )

    df.columns = df.columns.astype(str).str.strip()

    df["date"] = pd.to_datetime(
        df["DT"].astype(str).str.strip(),
        format="%m/%d/%y",
        errors="raise",
    )

    num_cols = ["ruo", "vol"]

    out = df[["date", *[c for c in num_cols if c in df.columns]]].copy()
    out = out.sort_values("date").reset_index(drop=True)

    rename_map = {
        "ruo": "macro_ru_ruonia_rate",
        "vol": "macro_ru_ruonia_vol"
    }

    out = out.rename(columns=rename_map)

    for c in rename_map.values():
        out[c] = pd.to_numeric(out[c].astype(str).str.replace(",", "."), errors="coerce")

    return out


def get_ruabitr(path):
    import pandas as pd

    df = pd.read_csv(
        path,
        sep=";",
        skiprows=2,
        header=0,
        decimal=",",
        encoding="cp1251",
    )

    df.columns = df.columns.astype(str).str.strip()

    df["date"] = pd.to_datetime(
        df["TRADEDATE"].astype(str).str.strip(),
        format="%d.%m.%Y",
        errors="raise",
    )

    rename_map = {
        "CLOSE": "bond_ru_corp_price_close",
        "YIELD": "bond_ru_corp_yield",
        "DURATION": "bond_ru_corp_duration"
    }

    actual_map = {old: new for old, new in rename_map.items() if old in df.columns}
    df = df.rename(columns=actual_map)

    value_cols = list(actual_map.values())
    out = df[["date", *value_cols]].copy()

    for c in value_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out.loc[out["bond_ru_corp_yield"] > 100, "bond_ru_corp_yield"] = pd.NA

    return out.sort_values("date").reset_index(drop=True)


def get_oil_brent_eia(path):
    import pandas as pd
    import re

    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    header_i = next(i for i, s in enumerate(lines) if s.lstrip().startswith("Date;"))

    df = pd.read_csv(
        path,
        sep=";",
        skiprows=header_i,
        header=0,
        encoding="utf-8",
        decimal=",",
        engine="python",
    )

    df = df.loc[:, ~df.columns.astype(str).str.contains("^Unnamed")].copy()

    months_map = {
        "янв.": 1, "февр.": 2, "марта": 3, "апр.": 4, "мая": 5, "июня": 6,
        "июля": 7, "авг.": 8, "сент.": 9, "окт.": 10, "нояб.": 11, "дек.": 12
    }

    def parse_russian_date(date_str):
        if not isinstance(date_str, str): return pd.NaT
        parts = re.findall(r'(\w+)\s+(\d+),\s+(\d{4})', date_str.lower())
        if parts:
            m_name, day, year = parts[0]
            month = months_map.get(m_name, '01')
            return f"{year}-{month}-{day.zfill(2)}"
        return pd.NaT

    df["date"] = pd.to_datetime(df["Date"].apply(parse_russian_date), errors="coerce")

    price_col = df.columns[1]
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")

    out = df[["date", price_col]].rename(columns={price_col: "brent_price"})
    out = out.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    return out


dict_func = {
             "Таблицы с рядами РК/Объем кредитования РК.csv": get_loans_RK,
             "Таблицы с рядами РК/Индексы потребительских цен г-г.csv": get_cpi_yoy_table,
             "Таблицы с рядами РК/Индексы государственных ценных бумаг.csv": get_zgb_cp,
             "Таблицы с рядами РК/Индекс цен корпоративных облигаций.csv": get_kase_bmc,
             "Таблицы с рядами РК/ВВП на душу населения.csv": get_gdp_per_capita_kz,
             "Таблицы с рядами РК/Базовая ставка РК.csv": get_base_rate_kz,
             "Таблицы с рядами РФ/Инфляция и ключевая ставка Банка России.csv": get_key_rate,
             "Таблицы с рядами РФ/Индекс потребительских цен.csv": get_price_idx,
             "Таблицы с рядами РФ/Бескупонная доходность ОФЗ.csv": get_yield_curve,
             "Таблицы с рядами РФ/Безработица.csv": get_labor_force,
             "Таблицы с рядами РФ/RUONIA(ставка овернайта).csv": get_ruonia,
             "Таблицы с рядами РФ/RUABITR(индекс корпоративных облигаций).csv": get_ruabitr,
             "в мире/Brent spot.csv": get_oil_brent_eia
}
# for path in dict_func:
#     seq = dict_func[path](path)
#     print(seq.shape)
#     print(seq.head(10))

