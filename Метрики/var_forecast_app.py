import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import joblib


@st.cache_resource(show_spinner=False)
def load_var_model(filepath='../var_ofz_model.pkl'):
    try:
        return joblib.load(filepath)
    except Exception as e:
        st.error(f"Ошибка загрузки VAR модели: {e}")
        return None


def render_var_forecast_block(df_st, df_raw):
    st.divider()
    st.subheader("Макро-прогноз кривой ОФЗ (VAR-модель на 1 день вперед)")

    var_model_results = load_var_model()
    if not var_model_results:
        return

    ofz_cols = [f'bond_ru_ofz_yield_{m}' for m in ['3m', '6m', '9m', '1y', '2y', '3y', '5y', '7y', '10y', '15y', '20y']]
    macro_cols = [
        'macro_ru_key_rate', 'macro_ru_ruonia_rate', 'macro_ru_inflation_yoy',
        'brent_price', 'bond_ru_corp_yield', 'macro_ru_labor_unemployment_rate', 'macro_kz_base_rate'
    ]

    missing_cols = [c for c in ofz_cols + macro_cols if c not in df_st.columns]
    if missing_cols:
        st.warning(f"Для VAR-прогноза не хватает колонок: {missing_cols}")
        return

    df_model = df_st[ofz_cols + macro_cols].copy()
    optimal_lags = {
        'macro_ru_key_rate': 23,
        'macro_kz_base_rate': 53,
        'macro_ru_ruonia_rate': 1,
        'macro_ru_inflation_yoy': 0,
        'brent_price': 0,
        'bond_ru_corp_yield': 0,
        'macro_ru_labor_unemployment_rate': 0
    }

    for col, lag in optimal_lags.items():
        if lag > 0:
            df_model[col] = df_model[col].shift(lag)

    df_model = df_model.dropna()

    if df_model.empty or len(df_model) < var_model_results.k_ar:
        st.info("Недостаточно исторических данных (с учетом лагов) для построения прогноза.")
        return

    k_ar = var_model_results.k_ar
    last_obs = df_model.values[-k_ar:]
    forecast_diff = var_model_results.forecast(last_obs, steps=1)
    delta_pred = forecast_diff[0][:len(ofz_cols)]

    last_known_date = df_raw[ofz_cols].dropna(how='all').index.max()
    current_levels = df_raw.loc[last_known_date, ofz_cols].values

    predicted_levels = current_levels + delta_pred

    tenors_labels = ['3m', '6m', '9m', '1y', '2y', '3y', '5y', '7y', '10y', '15y', '20y']

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
            line=dict(color='#d62728', width=3, dash='dash')
        ))

        fig.update_layout(
            title="Ожидаемый сдвиг кривой ОФЗ (На основе макро-факторов)",
            xaxis_title="Срок до погашения",
            yaxis_title="Доходность (%)",
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)