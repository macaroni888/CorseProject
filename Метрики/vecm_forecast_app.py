import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from statsmodels.tsa.vector_ar.vecm import VECM


def render_vecm_forecast_block(df_raw):
    st.divider()
    st.subheader("Коинтеграционный прогноз кривой ОФЗ (VECM на 1 день вперед)")

    ofz_core = [
        'bond_ru_ofz_yield_3m',
        'bond_ru_ofz_yield_9m',
        'bond_ru_ofz_yield_2y',
        'bond_ru_ofz_yield_5y',
        'bond_ru_ofz_yield_10y',
        'bond_ru_ofz_yield_20y'
    ]

    macro_anchors = [
        'bond_ru_corp_yield',
        'brent_price',
        'macro_ru_key_rate',
        'KASE_BMC',
        'macro_ru_ruonia_rate',
        'bond_kz_govt_cp_short_u1y'
    ]

    vecm_cols = ofz_core + macro_anchors

    missing_cols = [c for c in vecm_cols if c not in df_raw.columns]
    if missing_cols:
        st.warning(f"Для VECM-прогноза не хватает колонок: {missing_cols}")
        return

    df_model = df_raw[vecm_cols].dropna()

    if df_model.empty or len(df_model) < 50:
        st.info("Недостаточно исторических данных для обучения VECM.")
        return

    optimal_lag = 2
    coint_rank = 6

    try:
        vecm_model = VECM(df_model,
                          k_ar_diff=optimal_lag,
                          coint_rank=coint_rank,
                          deterministic="ci")
        vecm_res = vecm_model.fit()
        forecast_levels = vecm_res.predict(steps=1)

        delta_pred = forecast_levels[0][:len(ofz_core)] - df_model[ofz_core].iloc[-1].values

    except Exception as e:
        st.error(f"Ошибка расчета VECM: {e}")
        return

    last_known_date = df_model.index.max()
    current_levels = df_model[ofz_core].iloc[-1].values
    predicted_levels = current_levels + delta_pred

    tenors_labels = ['3m', '9m', '2y', '5y', '10y', '20y']

    plot_df = pd.DataFrame({
        'Tenor_Label': tenors_labels,
        'Current': current_levels,
        'Forecast': predicted_levels
    })

    col1, col2 = st.columns([1, 2])

    with col1:
        st.write(f"**Текущая дата:** {last_known_date.strftime('%Y-%m-%d')}")
        st.write("**Дельта прогноза (б.п.):**")

        plot_df['Expected_Change_bps'] = (plot_df['Forecast'] - plot_df['Current']) * 100
        display_df = plot_df[['Tenor_Label', 'Current', 'Forecast', 'Expected_Change_bps']].copy()
        display_df.columns = ['Срок', 'Текущая (%)', 'Прогноз (%)', 'Изменение (б.п.)']
        st.dataframe(display_df.style.format({
            'Текущая (%)': '{:.2f}', 'Прогноз (%)': '{:.2f}', 'Изменение (б.п.)': '{:+.1f}'
        }), hide_index=True)

    with col2:
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=plot_df['Tenor_Label'], y=plot_df['Current'],
            mode='lines+markers', name='Текущая кривая',
            line=dict(color='#1f77b4', width=3)
        ))

        fig.add_trace(go.Scatter(
            x=plot_df['Tenor_Label'], y=plot_df['Forecast'],
            mode='lines+markers', name='Прогноз на завтра',
            line=dict(color='#2ca02c', width=3, dash='dash')
        ))

        fig.update_layout(
            title="Ожидаемый сдвиг кривой ОФЗ (VECM: Ядро 12 переменных)",
            xaxis_title="Срок до погашения",
            yaxis_title="Доходность (%)",
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)