import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st


def render_portfolio_stress_test(df_raw, final_vecm_res):
    st.subheader("Сценарный Стресс-тест Портфеля")
    st.markdown("""
    Стресс-сценарии для портфеля. Алгоритм транслирует макро-шоки в деформацию кривой ОФЗ 
    и рассчитывает итоговое изменение стоимости портфеля.
    """)

    var_names = list(final_vecm_res.names)

    st.write("#### Структура портфеля и горизонт")
    col_p1, col_p2, col_p3, col_p4, col_p5 = st.columns([1.5, 1, 1, 1, 1])

    with col_p1:
        portfolio_size = st.number_input("Объем (млн руб.)", min_value=10, value=1000, step=100)
    with col_p2:
        horizon_days = st.number_input("Горизонт (дней)", min_value=5, max_value=90, value=30, step=5)
    with col_p3:
        w_3m = st.slider("ОФЗ 3m (Дюрация ~0.25)", 0, 100, 20)
    with col_p4:
        w_2y = st.slider("ОФЗ 2y (Дюрация ~1.8)", 0, 100, 30)
    with col_p5:
        w_10y = st.slider("ОФЗ 10y (Дюрация ~7.5)", 0, 100, 50)

    total_w = w_3m + w_2y + w_10y
    if total_w != 100:
        if total_w == 0:
            w_3m, w_2y, w_10y = 0.33, 0.33, 0.34
        else:
            w_3m, w_2y, w_10y = w_3m / total_w, w_2y / total_w, w_10y / total_w
    else:
        w_3m, w_2y, w_10y = w_3m / 100, w_2y / 100, w_10y / 100

    st.write("#### Векторы макро-шоков (Калибровка сценариев)")

    col_s1, col_s2, col_s3 = st.columns(3)
    col_s4, col_s5, col_s6 = st.columns(3)

    with col_s1:
        shock_key_bps = st.number_input("Шок ключевой ставки, б.п.", value=600, step=50)
    with col_s2:
        shock_brent = st.number_input("Шок цены на нефть Brent, $", value=-30.0, step=5.0)
    with col_s3:
        shock_corp_bps = st.number_input("Шок доходности корпоративных облигаций РФ (индекс RUABITR), б.п.", value=200,
                                         step=50)
    with col_s4:
        shock_kase_pct = st.number_input("Шок индекса KASE, %", value=-20.0, step=5.0)
    with col_s5:
        shock_ruonia_bps = st.number_input("Шок RUONIA, б.п.", value=300, step=50)
    with col_s6:
        shock_kz_bps = st.number_input("Шок доходности коротких гос. облигаций РК, б.п.", value=150, step=50)

    irf = final_vecm_res.irf(periods=horizon_days)
    residuals = final_vecm_res.resid
    current_kase = df_raw['KASE_BMC'].iloc[-1]

    scenarios_config = [
        {"name": "Шок ключевой ставки", "var": "macro_ru_key_rate", "shock": shock_key_bps / 100.0},
        {"name": "Шок цены на нефть", "var": "brent_price", "shock": shock_brent},
        {"name": "Шок на рынке корпоративных облигаций РФ", "var": "bond_ru_corp_yield",
         "shock": shock_corp_bps / 100.0},
        {"name": "Шок на рынке корпоративных облигаций РК", "var": "KASE_BMC",
         "shock": current_kase * (shock_kase_pct / 100.0)},
        {"name": "Шок ставки овернайта (RUONIA)", "var": "macro_ru_ruonia_rate", "shock": shock_ruonia_bps / 100.0},
        {"name": "Шок доходности коротких гос. облигаций РК", "var": "bond_kz_govt_cp_short_u1y",
         "shock": shock_kz_bps / 100.0}
    ]

    idx_3m = var_names.index('bond_ru_ofz_yield_3m')
    idx_2y = var_names.index('bond_ru_ofz_yield_2y')
    idx_10y = var_names.index('bond_ru_ofz_yield_10y')
    dur_3m, dur_2y, dur_10y = 0.25, 1.8, 7.5

    results = []
    curve_dynamics = {}

    for sc in scenarios_config:
        imp_var = sc["var"]
        if imp_var not in var_names:
            st.error(f"Внимание: переменная {imp_var} не найдена в модели!")
            continue

        imp_idx = var_names.index(imp_var)
        std_1_shock = np.std(residuals[:, imp_idx])
        scale_factor = sc["shock"] / std_1_shock if std_1_shock != 0 else 0

        scaled_irf_3m = irf.orth_irfs[:, idx_3m, imp_idx] * scale_factor
        scaled_irf_2y = irf.orth_irfs[:, idx_2y, imp_idx] * scale_factor
        scaled_irf_10y = irf.orth_irfs[:, idx_10y, imp_idx] * scale_factor

        dyn_dict = {
            '3m': scaled_irf_3m * 100,
            '2y': scaled_irf_2y * 100,
            '10y': scaled_irf_10y * 100
        }

        curve_dynamics[sc["name"]] = dyn_dict

        delta_3m = scaled_irf_3m[-1]
        delta_2y = scaled_irf_2y[-1]
        delta_10y = scaled_irf_10y[-1]

        pnl_3m = -dur_3m * (delta_3m / 100) * (portfolio_size * w_3m)
        pnl_2y = -dur_2y * (delta_2y / 100) * (portfolio_size * w_2y)
        pnl_10y = -dur_10y * (delta_10y / 100) * (portfolio_size * w_10y)
        total_pnl = pnl_3m + pnl_2y + pnl_10y

        results.append({
            "Сценарий": sc["name"],
            "Δ 3m (б.п.)": delta_3m * 100,
            "Δ 2y (б.п.)": delta_2y * 100,
            "Δ 10y (б.п.)": delta_10y * 100,
            "P&L (млн руб.)": total_pnl
        })

    df_results = pd.DataFrame(results)

    st.write(f"#### Финансовый результат (на {horizon_days}-й день)")
    col_heatmap, col_table = st.columns([1, 1.3])

    with col_heatmap:
        df_sorted = df_results.sort_values(by="P&L (млн руб.)", ascending=True)
        colors = ['#d62728' if val < 0 else '#2ca02c' for val in df_sorted["P&L (млн руб.)"]]

        fig_hm = go.Figure(go.Bar(
            x=df_sorted["P&L (млн руб.)"], y=df_sorted["Сценарий"],
            orientation='h', marker_color=colors,
            text=[f"{val:+.1f} млн" for val in df_sorted["P&L (млн руб.)"]],
            textposition='auto'
        ))
        fig_hm.update_layout(
            title="P&L портфеля",
            xaxis_title="Убыток / Прибыль ",
            margin=dict(l=10, r=10, t=30, b=10), height=300
        )
        st.plotly_chart(fig_hm, use_container_width=True)

    with col_table:
        st.markdown("<br>", unsafe_allow_html=True)
        st.dataframe(
            df_results.style.format({
                "Δ 3m (б.п.)": "{:+.1f}", "Δ 2y (б.п.)": "{:+.1f}",
                "Δ 10y (б.п.)": "{:+.1f}", "P&L (млн руб.)": "{:+.2f}"
            }).background_gradient(subset=["P&L (млн руб.)"], cmap="RdYlGn"),
            hide_index=True, use_container_width=True, height=300
        )

    st.write("#### Деформация кривой ОФЗ по дням")

    available_scenarios = [sc["name"] for sc in scenarios_config if sc["name"] != "Шок ключевой ставки"]
    default_idx = available_scenarios.index(
        "Шок ставки овернайта (RUONIA)") if "Шок ставки овернайта (RUONIA)" in available_scenarios else 0

    selected_scenario_2 = st.selectbox(
        "Выберите сценарий для детального просмотра (второй график):",
        options=available_scenarios,
        index=default_idx
    )

    days_x = np.arange(horizon_days + 1)

    def plot_curve_dynamics(fig, dyn_data, title):
        lines_config = [
            ('10y', '#1f77b4', 'rgba(31, 119, 180, 0.2)', '10y (Длинные)', 2),
            ('2y', '#ff7f0e', 'rgba(255, 127, 14, 0.2)', '2y (Средние)', 2),
            ('3m', '#d62728', 'rgba(214, 39, 40, 0.2)', '3m (Короткие)', 3)
        ]

        for tenor, color, fillcolor, name, width in lines_config:
            fig.add_trace(go.Scatter(
                x=days_x, y=dyn_data[tenor], mode='lines', name=name,
                line=dict(color=color, width=width)
            ))

        fig.add_hline(y=0, line_color='black', line_width=1.5)

        fig.update_layout(
            title=dict(text=title, font=dict(size=14)),
            xaxis_title="Дни после шока", yaxis_title="Δ Доходности (б.п.)",
            margin=dict(l=20, r=20, t=40, b=80),
            legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5, font=dict(size=11))
        )
        return fig

    col_g1, col_g2 = st.columns(2)

    with col_g1:
        sc_name_1 = "Шок ключевой ставки"
        if sc_name_1 in curve_dynamics:
            fig1 = go.Figure()
            plot_curve_dynamics(fig1, curve_dynamics[sc_name_1], f"{sc_name_1}")
            st.plotly_chart(fig1, use_container_width=True)

    with col_g2:
        if selected_scenario_2 in curve_dynamics:
            fig2 = go.Figure()
            plot_curve_dynamics(fig2, curve_dynamics[selected_scenario_2], f"{selected_scenario_2}")
            st.plotly_chart(fig2, use_container_width=True)