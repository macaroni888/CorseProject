import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

def render_macro_spreads_block(df_raw):
    st.divider()
    st.subheader("Глобальный мониторинг макро-спредов (РФ и Казахстан)")
    st.markdown("""
    Индикаторы структурного риска, основанные на коинтеграционных связях VECM. 
    Графики отображают стационарные спреды (Error Correction Terms), рассчитанные с учетом 
    весов долгосрочного равновесия $\\beta$. Выход осциллятора за красные границы (±1.5 Std) 
    означает макроэкономический дисбаланс и предвещает скорую коррекцию кривой ОФЗ.
    """)

    required_cols = [
        'macro_ru_ruonia_rate', 'macro_ru_key_rate',
        'bond_ru_corp_yield', 'bond_ru_corp_price_close',
        'bond_ru_ofz_yield_5y', 'bond_ru_ofz_yield_10y', 'bond_ru_ofz_yield_3m',
        'bond_kz_govt_cp_short_u1y', 'KASE_BMC', 'loans_kz_total'
    ]

    missing = [c for c in required_cols if c not in df_raw.columns]
    if missing:
        st.warning(f"Для блока спредов не хватает данных: {missing}")
        return

    df_metrics = pd.DataFrame(index=df_raw.index)

    df_metrics['ECT_3m_Liquidity'] = (
        df_raw['bond_ru_ofz_yield_3m']
        - 1.8049 * df_raw['macro_ru_ruonia_rate']
        + 0.8685 * df_raw['macro_ru_key_rate']
    ) * 100

    df_metrics['ECT_5y_Corp'] = (
        df_raw['bond_ru_ofz_yield_5y']
        - 1.0451 * df_raw['bond_ru_corp_yield']
        + 0.1269 * df_raw['macro_ru_key_rate']
    ) * 100

    df_metrics['ECT_10y_Macro'] = (
        df_raw['bond_ru_ofz_yield_10y']
        - 0.5992 * df_raw['bond_ru_corp_yield']
        + 0.3381 * df_raw['macro_ru_key_rate']
        - 0.2145 * df_raw['macro_ru_ruonia_rate']
    ) * 100

    last_date = df_raw.index[-1]

    df_monthly_loans = df_raw['loans_kz_total'].resample('ME').last().dropna()
    kz_credit_impulse = df_monthly_loans.pct_change() * 100
    kz_credit_impulse_plot = kz_credit_impulse.loc['2023-01-01':]

    def plot_standard_spread(series, title, y_label):
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        filtered_series = series[(series >= Q1 - 1.5 * IQR) & (series <= Q3 + 1.5 * IQR)]

        mean_val = filtered_series.mean()
        std_val = filtered_series.std()

        upper_bound = mean_val + 1.5 * std_val
        lower_bound = mean_val - 1.5 * std_val

        series_plot = series.loc['2023-01-01':]
        if series_plot.empty:
            return go.Figure(), 0, 0

        current_val = series_plot.iloc[-1]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=series_plot.index, y=series_plot, mode='lines', name='Спред',
                                 line=dict(color='#1f77b4', width=2)))
        fig.add_trace(go.Scatter(x=series_plot.index, y=[mean_val] * len(series_plot), mode='lines', name='Среднее',
                                 line=dict(color='gray', width=1, dash='dash')))
        fig.add_trace(go.Scatter(x=series_plot.index, y=[upper_bound] * len(series_plot), mode='lines', name='+1.5 Std',
                                 line=dict(color='red', width=1, dash='dot')))
        fig.add_trace(go.Scatter(x=series_plot.index, y=[lower_bound] * len(series_plot), mode='lines', name='-1.5 Std',
                                 line=dict(color='red', width=1, dash='dot')))

        fig.add_trace(go.Scatter(
            x=[last_date], y=[current_val], mode='markers+text', name='Сейчас',
            marker=dict(color='red', size=8), text=[f"{current_val:.0f}"], textposition="top center"
        ))

        fig.update_layout(title=title, yaxis_title=y_label, xaxis_title="", showlegend=False,
                          margin=dict(l=20, r=20, t=40, b=20), height=300)
        return fig, current_val, mean_val

    def plot_stat_arb_pair(series_Y, name_Y, series_X, name_X, title):
        df_clean = pd.concat([series_Y, series_X], axis=1).dropna()
        if df_clean.empty:
            return go.Figure()

        Y_clean = df_clean.iloc[:, 0]
        X_clean = df_clean.iloc[:, 1]

        beta, alpha = np.polyfit(X_clean, Y_clean, 1)
        synthetic_Y = beta * X_clean + alpha
        spread = Y_clean - synthetic_Y

        Q1 = spread.quantile(0.25)
        Q3 = spread.quantile(0.75)
        IQR = Q3 - Q1
        filtered_spread = spread[(spread >= Q1 - 1.5 * IQR) & (spread <= Q3 + 1.5 * IQR)]

        spread_mean = filtered_spread.mean()
        spread_std = filtered_spread.std()

        upper_bound = spread_mean + 1.5 * spread_std
        lower_bound = spread_mean - 1.5 * spread_std

        Y_plot = Y_clean.loc['2023-01-01':]
        X_plot = X_clean.loc['2023-01-01':]
        spread_plot = spread.loc['2023-01-01':]

        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            row_heights=[0.65, 0.35], vertical_spacing=0.08,
            specs=[[{"secondary_y": True}], [{"secondary_y": False}]]
        )

        fig.add_trace(
            go.Scatter(x=Y_plot.index, y=Y_plot, mode='lines', name=name_Y, line=dict(color='#1f77b4', width=2)), row=1,
            col=1, secondary_y=False)
        fig.add_trace(
            go.Scatter(x=X_plot.index, y=X_plot, mode='lines', name=name_X, line=dict(color='#ff7f0e', width=2)), row=1,
            col=1, secondary_y=True)

        fig.add_trace(go.Scatter(x=spread_plot.index, y=spread_plot, mode='lines', name='Спред',
                                 line=dict(color='purple', width=1.5)), row=2, col=1)
        fig.add_trace(go.Scatter(x=spread_plot.index, y=[spread_mean] * len(spread_plot), mode='lines', name='Баланс',
                                 line=dict(color='gray', width=1, dash='dash')), row=2, col=1)
        fig.add_trace(go.Scatter(x=spread_plot.index, y=[upper_bound] * len(spread_plot), mode='lines', name='+1.5 Std',
                                 line=dict(color='red', width=1, dash='dot')), row=2, col=1)
        fig.add_trace(go.Scatter(x=spread_plot.index, y=[lower_bound] * len(spread_plot), mode='lines', name='-1.5 Std',
                                 line=dict(color='red', width=1, dash='dot')), row=2, col=1)

        fig.update_layout(title=title, margin=dict(l=20, r=20, t=40, b=10), height=350, showlegend=False,
                          hovermode="x unified")
        fig.update_yaxes(title_text=name_Y, secondary_y=False, row=1, col=1, title_font=dict(size=10))
        fig.update_yaxes(title_text=name_X, secondary_y=True, row=1, col=1, title_font=dict(size=10))
        fig.update_yaxes(title_text="Отклонение", secondary_y=False, row=2, col=1, title_font=dict(size=10))

        return fig

    col1, col2, col3 = st.columns(3)

    with col1:
        fig1, curr1, mean1 = plot_standard_spread(df_metrics['ECT_3m_Liquidity'], "ОФЗ_3М - 1.80×RUONIA + 0.87×Ключ.ставка", "б.п.")
        st.plotly_chart(fig1, use_container_width=True)
        st.caption(f"Отклонение от баланса: **{curr1 - mean1:.0f} б.п.**")

    with col2:
        fig2, curr2, mean2 = plot_standard_spread(df_metrics['ECT_5y_Corp'], "ОФЗ_5Л - 1.05×Корп.доходность + 0.13×Ключ.ставка", "б.п.")
        st.plotly_chart(fig2, use_container_width=True)
        st.caption(f"Отклонение от баланса: **{curr2 - mean2:.0f} б.п.**")

    with col3:
        fig3, curr3, mean3 = plot_standard_spread(df_metrics['ECT_10y_Macro'], "ОФЗ_10Л - 0.60×Корп.дох + 0.34×Ключ.ст - 0.21×RUONIA", "б.п.")
        st.plotly_chart(fig3, use_container_width=True)
        st.caption(f"Отклонение от баланса: **{curr3 - mean3:.0f} б.п.**")

    st.write("")

    col4, col5, col6 = st.columns(3)

    with col4:
        fig4 = plot_stat_arb_pair(
            df_raw['bond_kz_govt_cp_short_u1y'], 'Индекс KZGB_CPs',
            df_raw['bond_ru_corp_price_close'], 'Цена Corp РФ',
            "Коинтеграция: KZGB_CPs vs Цена Corp РФ"
        )
        st.plotly_chart(fig4, use_container_width=True)
        st.caption("Снизу — спред отклонений от баланса.")

    with col5:
        fig5 = plot_stat_arb_pair(
            df_raw['KASE_BMC'], 'Индекс KASE',
            df_raw['bond_ru_corp_price_close'], 'Цена Corp РФ',
            "Коинтеграция: Индекс KASE vs Цена корп. индекса РФ"
        )
        st.plotly_chart(fig5, use_container_width=True)
        st.caption("Снизу — спред отклонений от баланса.")

    with col6:
        fig6 = go.Figure()
        colors = ['#2ca02c' if val >= 0 else '#d62728' for val in kz_credit_impulse_plot]

        fig6.add_trace(go.Bar(
            x=kz_credit_impulse_plot.index,
            y=kz_credit_impulse_plot,
            marker_color=colors,
            name="Импульс"
        ))

        fig6.update_layout(
            title="Кредитный Импульс РК (М/М)",
            yaxis_title="Прирост кредитов, %",
            margin=dict(l=20, r=20, t=40, b=20),
            height=350,
            showlegend=False
        )

        st.plotly_chart(fig6, use_container_width=True)

        last_impulse = kz_credit_impulse_plot.iloc[-1] if not kz_credit_impulse_plot.empty else 0
        st.caption(f"Последний месяц: **{last_impulse:.2f}%**")