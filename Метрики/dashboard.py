import joblib
from datetime import datetime
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(layout="wide", page_title="Спред-аналитика РФ и РК")

from read_functions import dict_func
from correlation_matrixes import render_lagged_correlation_dashboard
from ofz_forecast import render_ofz_forecast_block
from var_forecast_app import render_var_forecast_block
from vecm_forecast_app import render_vecm_forecast_block
from macro_spreads_app import render_macro_spreads_block
from stress_tests import render_portfolio_stress_test


def create_master_df(dict_func):
    all_series = []
    for path, func in dict_func.items():
        try:
            df = func(path)
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
            all_series.append(df)
        except Exception as e:
            print(f"Ошибка в {path}: {e}")

    master = pd.concat(all_series, axis=1)
    master = master.sort_index().ffill()
    master = master.loc['2020-01-01':'2026-01-01']
    return master


@st.cache_data
def load_data():
    return create_master_df(dict_func)


@st.cache_data
def load_stationary_data():
    return pd.read_csv('master_df_daily_stationary.csv', index_col=0, parse_dates=True)


@st.cache_resource
def load_model():
    try:
        model = joblib.load('../vecm_ofz_model.pkl')
        return model
    except FileNotFoundError:
        return None


df = load_data()
df_st = load_stationary_data()
final_vecm_res = load_model()

st.title("Мониторинг макроиндикаторов и долговых рынков")

tab1, tab2, tab3, tab4 = st.tabs([
    "Аналитика и Корреляции",
    "Макро-спреды (VECM)",
    "Прогнозирование",
    "Стресс-тестирование"
])

with tab1:
    st.write("##### Настройки аналитики")
    col_ctrl1, col_ctrl2 = st.columns(3)[:2]

    with col_ctrl1:
        date_range = st.date_input(
            "Диапазон дат",
            value=(df.index.min(), df.index.max())
        )
    with col_ctrl2:
        selected_metrics = st.multiselect(
            "Выберите показатели для сравнения:",
            options=df.columns,
            default=["brent_price"] if "brent_price" in df.columns else [df.columns[0]]
        )

    st.divider()

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Динамика показателей")

        if len(date_range) == 2:
            mask = (df.index >= pd.Timestamp(date_range[0])) & (df.index <= pd.Timestamp(date_range[1]))
            filtered_df = df.loc[mask, selected_metrics]

            if st.checkbox("Нормализовать данные (0-1)"):
                filtered_df = (filtered_df - filtered_df.min()) / (filtered_df.max() - filtered_df.min())

            fig = px.line(filtered_df, x=filtered_df.index, y=filtered_df.columns)

            fig.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Пожалуйста, выберите начальную и конечную дату.")

    with col2:
        st.subheader("Матрица корреляций (Стационарная)")
        if len(date_range) == 2:
            mask_st = (df_st.index >= pd.Timestamp(date_range[0])) & (df_st.index <= pd.Timestamp(date_range[1]))
            filtered_df_st = df_st.loc[mask_st]

            if len(selected_metrics) > 1:
                valid_metrics_st = [m for m in selected_metrics if m in filtered_df_st.columns]
                if valid_metrics_st:
                    corr = filtered_df_st[valid_metrics_st].corr()
                    st.dataframe(corr.style.background_gradient(cmap='coolwarm'), use_container_width=True)
                else:
                    st.info("Выбранные метрики отсутствуют в стационарном датасете.")
            else:
                st.info("Выберите хотя бы 2 показателя для расчета корреляции.")

    st.divider()
    if len(date_range) == 2:
        render_lagged_correlation_dashboard(filtered_df_st)

with tab2:
    st.markdown("В этом разделе отслеживаются спреды на основе коинтеграционных уравнений VECM.")
    render_macro_spreads_block(df)

with tab3:
    st.markdown("Сравнение краткосрочных прогнозов кривой ОФЗ от разных эконометрических моделей.")
    with st.expander("Базовый прогноз (ARIMA/Naive)", expanded=True):
        render_ofz_forecast_block(df)
    with st.expander("Векторная авторегрессия (VAR)", expanded=True):
        render_var_forecast_block(df_st, df)
    with st.expander("Модель коррекции ошибок (VECM)", expanded=True):
        render_vecm_forecast_block(df)

with tab4:
    if final_vecm_res is not None:
        render_portfolio_stress_test(df, final_vecm_res)
    else:
        st.error("Модель VECM не найдена! Обучите и сохраните 'vecm_ofz_model.pkl'.")