import pandas as pd
import streamlit as st


def _render_single_corr_block(df, block_id, default_target_blocks=None, default_metric_blocks=None):
    st.markdown(f"**Матрица кросс-корреляций #{block_id}**")

    blocks_definition = {
        "ОФЗ РФ": [c for c in df.columns if c.startswith("bond_ru_ofz_yield_")],
        "Макро РФ": [c for c in df.columns if
                                        c in ["macro_ru_key_rate", "macro_ru_inflation_yoy"] or c.startswith(
                                            "macro_ru_cpi_")],
        "Кредитный рынок РФ": [c for c in df.columns if
                                              c.startswith("macro_ru_ruonia_") or c.startswith("bond_ru_corp_")],
        "Рынок труда РФ": [c for c in df.columns if c.startswith("macro_ru_labor_")],

        "ГЦБ РК + индекс KASE": [c for c in df.columns if c.startswith("bond_kz_govt_cp_") or c == "KASE_BMC"],
        "Макро РК": [c for c in df.columns if
                                         c in ["macro_kz_base_rate", "macro_kz_usd_kzt_avg"] or c.startswith(
                                             "macro_kz_gdp_")],
        "Инфляция РК": [c for c in df.columns if c.startswith("cpi_kz_")],
        # "Кредитование РК": [c for c in df.columns if c.startswith("loans_kz_")],

        "Глобальные (Нефть)": [c for c in df.columns if c == "brent_price"],
    }

    available_blocks = {k: v for k, v in blocks_definition.items() if len(v) > 0}
    block_names = list(available_blocks.keys())

    valid_def_targets = [b for b in (default_target_blocks or []) if b in block_names]
    target_blocks = st.multiselect(
        "Таргеты (строки):",
        options=block_names,
        default=valid_def_targets if valid_def_targets else ([block_names[0]] if block_names else None),
        key=f"target_blocks_{block_id}"
    )

    valid_def_metrics = [b for b in (default_metric_blocks or []) if b in block_names]
    metric_blocks = st.multiselect(
        "Метрики (столбцы):",
        options=block_names,
        default=valid_def_metrics if valid_def_metrics else ([block_names[1]] if len(block_names) > 1 else None),
        key=f"metric_blocks_{block_id}"
    )

    lag_range = st.slider("Диапазон поиска лагов (в днях):", -60, 60, (-30, 30), key=f"lags_{block_id}")

    targets = []
    for b in target_blocks: targets.extend(available_blocks[b])
    metrics = []
    for b in metric_blocks: metrics.extend(available_blocks[b])
    targets, metrics = list(dict.fromkeys(targets)), list(dict.fromkeys(metrics))

    if targets and metrics:
        lags = list(range(lag_range[0], lag_range[1] + 1))
        corr_matrix = pd.DataFrame(index=targets, columns=metrics)
        lags_matrix = pd.DataFrame(index=targets, columns=metrics)

        for t in targets:
            for m in metrics:
                best_lag, max_abs_corr, best_corr_val = None, -1.0, 0.0
                for lag in lags:
                    shifted_target = df[t].shift(-lag)
                    corr_val = shifted_target.corr(df[m])
                    if pd.notna(corr_val) and abs(corr_val) > max_abs_corr:
                        max_abs_corr, best_corr_val, best_lag = abs(corr_val), corr_val, lag
                corr_matrix.loc[t, m], lags_matrix.loc[t, m] = best_corr_val, best_lag

        def clean_name(name):
            prefixes_to_remove = [
                "bond_ru_ofz_yield_", "bond_kz_govt_cp_", "macro_ru_",
                "macro_kz_", "cpi_kz_eoy_", "loans_kz_", "bond_ru_"
            ]
            clean = name
            for p in prefixes_to_remove:
                if clean.startswith(p):
                    clean = clean.replace(p, "")
            return clean

        display_corr = corr_matrix.copy()
        display_corr.index = [clean_name(i) for i in display_corr.index]
        display_corr.columns = [clean_name(c) for c in display_corr.columns]

        display_lags = lags_matrix.copy()
        display_lags.index = [clean_name(i) for i in display_lags.index]
        display_lags.columns = [clean_name(c) for c in display_lags.columns]

        styles = [
            dict(selector="th.col_heading", props=[
                ("writing-mode", "vertical-rl"),
                ("vertical-align", "middle"),
                ("height", "140px"),
                ("min-width", "30px"),
                ("font-size", "11px"),
                ("padding", "4px")
            ]),
            dict(selector="th.row_heading", props=[
                ("text-align", "left"),
                ("font-size", "11px"),
                ("min-width", "100px")
            ]),
            dict(selector="table", props=[
                ("white-space", "nowrap")
            ])
        ]

        st.markdown("**Максимальная корреляция:**")
        styled_corr = display_corr.astype(float).style.background_gradient(
            cmap='coolwarm', axis=None, vmin=-1, vmax=1
        ).format("{:.2f}").set_table_styles(styles)

        html_corr = f"<div style='width: 100%; overflow-x: auto;'>{styled_corr.to_html()}</div>"
        st.write(html_corr, unsafe_allow_html=True)

        with st.expander("Матрица оптимальных лагов (в днях)"):
            styled_lags = display_lags.astype(float).style.set_table_styles(styles).format("{:.0f}")
            html_lags = f"<div style='width: 100%; overflow-x: auto;'>{styled_lags.to_html()}</div>"
            st.write(html_lags, unsafe_allow_html=True)

    elif target_blocks or metric_blocks:
        st.warning("Выберите блоки таргетов и метрик.")


def render_lagged_correlation_dashboard(df):
    st.header("Оптимальные кросс-корреляции (по группам)")
    numeric_df = df.select_dtypes(include=['float64', 'int64'])

    if numeric_df.empty:
        st.error("В данных нет числовых столбцов.")
        return

    row1_col1, row1_col2 = st.columns([2, 1])
    row2_col1, row2_col2 = st.columns([2, 1])

    with row1_col1:
        _render_single_corr_block(
            numeric_df, 1,
            default_target_blocks=["ОФЗ РФ"],
            default_metric_blocks=["Глобальные (Нефть)", "Макро РК"]
        )

    with row1_col2:
        _render_single_corr_block(
            numeric_df, 2,
            default_target_blocks=["Кредитный рынок РФ"],
            default_metric_blocks=["ГЦБ РК + индекс KASE"]
        )

    st.divider()

    with row2_col1:
        _render_single_corr_block(
            numeric_df, 3,
            default_target_blocks=["ОФЗ РФ"],
            default_metric_blocks=["Макро РФ", "Инфляция РК"]
        )

    with row2_col2:
        _render_single_corr_block(
            numeric_df, 4,
            default_target_blocks=["Макро РФ"],
            default_metric_blocks=["ГЦБ РК + индекс KASE"]
        )