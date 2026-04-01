import pandas as pd
import numpy as np
import joblib
import streamlit as st
import plotly.graph_objects as go
import copy


@st.cache_resource(show_spinner=False)
def load_production_models(filepath="../ofz_arima_models.pkl"):
    try:
        models = joblib.load(filepath)
        return models
    except Exception as e:
        st.error(f"Ошибка загрузки моделей: {e}")
        return None


def get_dashboard_forecast(models_dict, df):
    n_periods = 1
    forecasts = []

    for col, model_data in models_dict.items():
        actual_model = copy.deepcopy(model_data['model'])
        last_train_date = model_data.get('last_train_date', pd.to_datetime('2020-01-01'))

        new_data = df[col].loc[df.index > last_train_date].dropna()

        if not new_data.empty:
            actual_model.update(new_data)

        pred = actual_model.predict(n_periods=n_periods)
        final_pred = pred.iloc[-1] if isinstance(pred, pd.Series) else pred[-1]

        forecasts.append({
            'Tenor': col.replace('bond_ru_ofz_yield_', ''),
            'Forecast_Yield': round(final_pred, 2)
        })

    return pd.DataFrame(forecasts)


def render_ofz_forecast_block(df):
    st.divider()
    st.subheader("Инерционный прогноз кривой ОФЗ (ARIMA на 1 день вперед)")

    ofz_cols = [c for c in df.columns if c.startswith('bond_ru_ofz_yield')]

    if not ofz_cols:
        st.info("Ряды доходностей ОФЗ не найдены для построения прогноза.")
        return

    models_dict = load_production_models()

    if models_dict:
        last_known_date = df[ofz_cols].dropna(how='all').index.max()

        f_col1, f_col2 = st.columns([1, 2])

        with f_col1:
            st.write(f"**Текущая дата:** {last_known_date.strftime('%Y-%m-%d')}")
            st.write("**Дельта прогноза (б.п.):**")

            forecast_df = get_dashboard_forecast(models_dict, df)

            current_curve = []
            for col in ofz_cols:
                tenor = col.replace('bond_ru_ofz_yield_', '')
                val = df[col].dropna().iloc[-1]
                current_curve.append({'Tenor': tenor, 'Current_Yield': round(val, 2)})

            current_df = pd.DataFrame(current_curve)

            if forecast_df is not None:
                compare_df = pd.merge(current_df, forecast_df, on='Tenor')

                tenor_order = {'3m': 0.25, '6m': 0.5, '9m': 0.75, '1y': 1, '2y': 2, '3y': 3, '5y': 5, '7y': 7,
                               '10y': 10, '15y': 15, '20y': 20}
                compare_df['sort_key'] = compare_df['Tenor'].map(tenor_order)
                compare_df = compare_df.sort_values('sort_key').drop(columns=['sort_key'])

                compare_df['Expected_Change_bps'] = (compare_df['Forecast_Yield'] - compare_df['Current_Yield']) * 100

                display_df = compare_df[['Tenor', 'Current_Yield', 'Forecast_Yield', 'Expected_Change_bps']].copy()
                display_df.columns = ['Срок', 'Текущая (%)', 'Прогноз (%)', 'Изменение (б.п.)']

                st.dataframe(display_df.style.format({
                    'Текущая (%)': '{:.2f}', 'Прогноз (%)': '{:.2f}', 'Изменение (б.п.)': '{:+.1f}'
                }), hide_index=True)

        with f_col2:
            if forecast_df is not None:
                fig_curve = go.Figure()

                fig_curve.add_trace(go.Scatter(
                    x=compare_df['Tenor'], y=compare_df['Current_Yield'],
                    mode='lines+markers', name='Текущая кривая',
                    line=dict(color='#1f77b4', width=3)
                ))

                fig_curve.add_trace(go.Scatter(
                    x=compare_df['Tenor'], y=compare_df['Forecast_Yield'],
                    mode='lines+markers', name='Прогноз на завтра',
                    line=dict(color='#d62728', width=3, dash='dash')
                ))

                fig_curve.update_layout(
                    title="Ожидаемый сдвиг кривой ОФЗ (На основе моментума)",
                    xaxis_title="Срок до погашения",
                    yaxis_title="Доходность (%)",
                    hovermode="x unified"
                )

                st.plotly_chart(fig_curve, use_container_width=True)