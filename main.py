import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
import warnings
import io

warnings.filterwarnings('ignore')

# Настройка страницы
st.set_page_config(
    page_title="🌍 Глобальный Аналитический Дашборд Стран Мира",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Кастомные стили
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        border: 1px solid #e6e9ef;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
    }
    .big-font {
        font-size: 2rem !important;
        font-weight: bold;
        color: #1f77b4;
    }
    .highlight {
        background-color: #ffeaa7;
        padding: 0.5rem;
        border-radius: 0.3rem;
        border-left: 4px solid #fdcb6e;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Загрузка и предобработка данных"""
    df = pd.read_csv(
        'world-data-2023.csv',
        thousands=',',
        decimal='.',
        encoding='utf-8'
    )

    # Очистка названий колонок
    df.columns = df.columns.str.strip().str.replace('\n', ' ').str.replace('\r', '')

    # --- Исправление названий колонок для соответствия списку фичей ---
    # Этот блок кода гарантирует, что названия колонок в DataFrame
    # точно соответствуют ожидаемым в вашем списке фичей.
    df.rename(columns={
        'Agricultural Land( %)': 'Agricultural Land (%)',
        'Land Area(Km2)': 'Land Area (Km2)',
        'Armed Forces size': 'Armed Forces Size',
        'Gasoline Price': 'Gasoline Price',  # Оставляем 'Gasoline Price' если это имя в исходных данных
        'Gross primary education enrollment (%)': 'Gross Primary Education Enrollment (%)',
        'Gross tertiary education enrollment (%)': 'Gross Tertiary Education Enrollment (%)',
        'Infant mortality': 'Infant Mortality',
        'Maternal mortality ratio': 'Maternal Mortality Ratio',
        'Minimum wage': 'Minimum Wage',
        'Out of pocket health expenditure': 'Out of pocket health expenditure',
        # Оставляем без (%) если это имя в исходных данных
        'Physicians per thousand': 'Physicians per thousand',
        'Population: Labor force participation (%)': 'Population: Labor force participation (%)',
        'Tax revenue (%)': 'Tax revenue (%)',
        'Unemployment rate': 'Unemployment rate',
        'Urban_population': 'Urban_population'  # Оставляем 'Urban_population' если это имя в исходных данных
    }, inplace=True)

    # Проверка и корректировка названий, используемых в numeric_columns
    # для обеспечения соответствия.
    # В этом месте вы должны решить, какое название (из вашего списка фичей или из numeric_columns)
    # является правильным. Я привел numeric_columns к вашему списку фичей.
    numeric_columns = [
        'Density (P/Km2)', 'Agricultural Land (%)', 'Land Area (Km2)',
        'Armed Forces Size', 'Birth Rate', 'Co2-Emissions', 'CPI',
        'CPI Change (%)', 'Fertility Rate', 'Forested Area (%)',
        'Gasoline Price',  # Изменено с 'Gasoline_Price'
        'GDP', 'Gross Primary Education Enrollment (%)',
        'Gross Tertiary Education Enrollment (%)', 'Infant Mortality',
        'Life expectancy', 'Maternal Mortality Ratio', 'Minimum Wage',
        'Out of pocket health expenditure',  # Изменено с 'Out of Pocket Health Expenditure (%)'
        'Physicians per thousand',  # Изменено с 'Physicians per Thousand'
        'Population', 'Population: Labor force participation (%)',
        # Изменено с 'Population: Labor Force Participation (%)'
        'Tax revenue (%)',  # Изменено с 'Tax Revenue (%)'
        'Total tax rate', 'Unemployment rate',  # Изменено с 'Unemployment Rate'
        'Urban_population',  # Изменено с 'Urban Population'
        'Latitude', 'Longitude'
    ]

    # Конвертация 'Life expectancy' и 'Co2-Emissions'
    df['Life expectancy'] = (
        df['Life expectancy']
        .astype(str)
        .str.replace(',', '.', regex=False)
        .replace('None', np.nan)
        .astype(float)
    )

    df['Land Area (Km2)'] = (
        df['Land Area (Km2)']
        .astype(str)
        .str.replace(',', '.', regex=False)
        .replace('None', np.nan)
        .astype(float)
    )

    # Очищаем и конвертим в числа
    for col in numeric_columns:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(r'[^0-9\.\-]', '', regex=True)
                .replace('', np.nan)
            )
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            st.warning(f"Колонка '{col}' не найдена в DataFrame после переименования. Проверьте исходные данные.")

    # # ---- отладочная проверка, можно потом убрать ----
    # st.write("Пропущено в Population:", df['Population'].isna().sum(), "из", len(df))
    # st.write("Названия колонок после загрузки и исправления:", df.columns.tolist())
    # # -----------------------------------------------

    return df


def safe_column_check(df, column_name):
    """Безопасная проверка существования колонки"""
    return column_name in df.columns and not df[column_name].isna().all()


def create_metric_cards(df, selected_countries):
    """Создание карточек с ключевыми метриками"""
    try:
        filtered_df = df[df['Country'].isin(selected_countries)] if selected_countries else df

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            # ИСПРАВЛЕНИЕ: Безопасное вычисление суммы с обработкой NaN
            total_pop = filtered_df['Population'].fillna(0).sum() if 'Population' in filtered_df.columns else 0
            st.markdown("""
            <div class="metric-card">
                <h3>👥 Общая Популяция</h3>
                <h2>{:,.0f}</h2>
                <p>человек</p>
            </div>
            """.format(total_pop), unsafe_allow_html=True)

        with col2:
            # ИСПРАВЛЕНИЕ: Безопасное вычисление среднего с обработкой NaN
            if 'GDP' in filtered_df.columns and not filtered_df['GDP'].isna().all():
                avg_gdp = filtered_df['GDP'].dropna().mean()
            else:
                avg_gdp = 0
            st.markdown("""
            <div class="metric-card">
                <h3>💰 Средний ВВП</h3>
                <h2>${:,.0f}</h2>
                <p>миллиардов</p>
            </div>
            """.format(avg_gdp / 1e9 if avg_gdp > 0 else 0), unsafe_allow_html=True)

        with col3:
            # ИСПРАВЛЕНИЕ: Безопасное вычисление среднего с обработкой NaN
            if 'Life expectancy' in filtered_df.columns and not filtered_df['Life expectancy'].isna().all():
                avg_life_exp = filtered_df['Life expectancy'].dropna().mean()
            else:
                avg_life_exp = 0
            st.markdown("""
            <div class="metric-card">
                <h3>🏥 Ср. Ожидаемость Жизни</h3>
                <h2>{:.1f}</h2>
                <p>лет</p>
            </div>
            """.format(avg_life_exp), unsafe_allow_html=True)

        with col4:
            # Безопасное вычисление среднего с обработкой NaN
            if 'Land Area (Km2)' in filtered_df.columns and not filtered_df['Land Area (Km2)'].isna().all():
                avg_area = filtered_df['Land Area (Km2)'].dropna().mean()
            else:
                avg_area = 0

            # Форматирование средней площади
            if avg_area >= 1e6:
                display_area = f"{avg_area / 1e6:.2f}M"
            elif avg_area >= 1e3:
                display_area = f"{avg_area / 1e3:.2f}K"
            else:
                display_area = f"{avg_area:.2f}"

            st.markdown(f"""
            <div class="metric-card">
                <h3>🌍 Ср. Площадь Суши</h3>
                <h2>{display_area}</h2>
                <p>км²</p>
            </div>
            """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Ошибка при создании карточек метрик: {str(e)}")
        st.info("Проверьте корректность данных")


def create_correlation_heatmap(df, selected_countries):
    """Создание тепловой карты корреляций"""
    try:
        filtered_df = df[df['Country'].isin(selected_countries)] if selected_countries else df

        # Если выбрано менее 3 стран, показываем корреляции по всем странам
        if selected_countries and len(selected_countries) < 3:
            st.info(
                f"ℹ️ Для корреляционного анализа используются все страны (выбрано только {len(selected_countries)} стран)")
            filtered_df = df

        # Используем только существующие колонки
        available_numeric_cols = []
        potential_cols = ['Population', 'GDP', 'Life expectancy', 'Birth Rate', 'Unemployment rate',
                          'Urban_population', 'Co2-Emissions', 'Physicians per thousand']

        for col in potential_cols:
            if safe_column_check(filtered_df, col):
                available_numeric_cols.append(col)

        if len(available_numeric_cols) < 2:
            st.warning("Недостаточно числовых данных для построения корреляционной матрицы")
            return None

        corr_data = filtered_df[available_numeric_cols].corr()

        fig = px.imshow(
            corr_data,
            labels=dict(x="Показатели", y="Показатели", color="Корреляция"),
            x=corr_data.columns,
            y=corr_data.columns,
            color_continuous_scale='RdBu_r',
            aspect="auto",
            title="🔥 Тепловая карта корреляций между показателями"
        )

        fig.update_layout(
            title_font=dict(size=20, color='#2c3e50'),
            height=600,
            font=dict(size=12)
        )

        return fig
    except Exception as e:
        st.error(f"Ошибка при создании тепловой карты: {str(e)}")
        return None


def create_gdp_vs_life_expectancy(df, selected_countries):
    """Создание scatter plot ВВП vs Ожидаемая продолжительность жизни"""
    try:
        if selected_countries and len(selected_countries) <= 2:
            filtered_df = df.copy()
            filtered_df['Выделено'] = filtered_df['Country'].apply(
                lambda x: 'Выбранная страна' if x in selected_countries else 'Другие страны'
            )
            title = f"💎 ВВП vs Ожидаемая продолжительность жизни"
            color_col = 'Выделено'
            color_map = {'Выбранная страна': 'red', 'Другие страны': 'lightblue'}
        else:
            filtered_df = df[df['Country'].isin(selected_countries)] if selected_countries else df
            filtered_df['Выделено'] = 'Все страны'
            title = "💎 ВВП vs Ожидаемая продолжительность жизни"
            color_col = 'Urban_population' if safe_column_check(filtered_df, 'Urban_population') else None
            color_map = None

        # Проверяем наличие необходимых колонок
        required_cols = ['GDP', 'Life expectancy', 'Population', 'Country']
        missing_cols = [col for col in required_cols if not safe_column_check(filtered_df, col)]

        if missing_cols:
            st.warning(f"Отсутствуют данные для колонок: {', '.join(missing_cols)}")
            return None

        # ИСПРАВЛЕНИЕ: Удаляем строки с NaN в ключевых колонках
        filtered_df = filtered_df.dropna(subset=['GDP', 'Life expectancy', 'Population'])

        if filtered_df.empty:
            st.warning("Нет данных для отображения после удаления пустых значений")
            return None

        hover_data = {}
        if safe_column_check(filtered_df, 'Birth Rate'):
            hover_data['Birth Rate'] = True
        if safe_column_check(filtered_df, 'Unemployment rate'):
            hover_data['Unemployment rate'] = True

        # Минимальный и максимальный радиус круга
        MIN_SIZE = 8
        MAX_SIZE = 60

        # ИСПРАВЛЕНИЕ: Безопасное нормализованное значение размера с обработкой NaN
        pop = filtered_df['Population'].astype(float)
        pop_min = pop.min()
        pop_max = pop.max()

        # Избегаем деления на ноль и обрабатываем NaN
        def scale_population(val):
            if pd.isna(val):
                return (MIN_SIZE + MAX_SIZE) / 2
            if pop_max == pop_min or pop_max == 0:
                return (MIN_SIZE + MAX_SIZE) / 2
            return MIN_SIZE + (val - pop_min) / (pop_max - pop_min) * (MAX_SIZE - MIN_SIZE)

        filtered_df['_BubbleSize'] = pop.apply(scale_population)

        # ИСПРАВЛЕНИЕ: Дополнительная проверка на NaN в размерах
        filtered_df['_BubbleSize'] = filtered_df['_BubbleSize'].fillna((MIN_SIZE + MAX_SIZE) / 2)

        fig = px.scatter(
            filtered_df,
            x='GDP',
            y='Life expectancy',
            size='_BubbleSize',
            color=color_col,
            hover_name='Country',
            hover_data=hover_data,
            title=title,
            labels={
                'GDP': 'ВВП (USD)',
                'Life expectancy': 'Ожидаемая продолжительность жизни (лет)',
                'Population': 'Население',
                'Urban_population': 'Городское население (%)',
                'Выделено': 'Категория'
            },
            color_continuous_scale='viridis' if not color_map else None,
            color_discrete_map=color_map
        )

        fig.update_layout(
            title_font=dict(size=20, color='#2c3e50'),
            height=600,
            showlegend=True
        )

        fig.update_traces(marker=dict(line=dict(width=1, color='white')))

        return fig
    except Exception as e:
        st.error(f"Ошибка при создании scatter plot: {str(e)}")
        return None


def create_population_pyramid(df, selected_countries):
    """Создание диаграммы населения по странам"""
    try:
        if selected_countries and len(selected_countries) == 1:
            # Для одной страны показываем сравнение с топ-10 странами
            country_name = selected_countries[0]
            selected_country_data = df[df['Country'] == country_name]
            top_countries = df.nlargest(10, 'Population')

            # Объединяем данные выбранной страны с топ-10
            if not selected_country_data.empty and country_name not in top_countries['Country'].values:
                display_df = pd.concat([top_countries, selected_country_data])
            else:
                display_df = top_countries

            title = f"🏙️ Население: {country_name} в сравнении с крупнейшими странами"

            # Выделяем выбранную страну цветом
            colors = ['red' if country == country_name else 'lightblue' for country in display_df['Country']]

        else:
            if selected_countries:
                filtered_df = df[df['Country'].isin(selected_countries)]
            else:
                filtered_df = df.dropna(subset=['Population']).nlargest(15, 'Population')

            display_df = filtered_df.sort_values('Population', ascending=True)
            title = "🏙️ Население по странам (Топ-15)"
            colors = 'Population'

        if not safe_column_check(display_df, 'Population'):
            st.warning("Отсутствуют данные о населении")
            return None

        fig = px.bar(
            display_df.sort_values('Population', ascending=True),
            x='Population',
            y='Country',
            orientation='h',
            title=title,
            labels={'Population': 'Население', 'Country': 'Страна'},
            color=colors,
            color_continuous_scale='plasma' if colors == 'Population' else None,
            color_discrete_map={country: color for country, color in zip(display_df['Country'], colors)} if isinstance(
                colors, list) else None
        )

        fig.update_layout(
            title_font=dict(size=20, color='#2c3e50'),
            height=600,
            yaxis={'categoryorder': 'total ascending'}
        )

        return fig
    except Exception as e:
        st.error(f"Ошибка при создании диаграммы населения: {str(e)}")
        return None


def create_multi_indicator_radar(df, selected_countries):
    """Создание радарной диаграммы для сравнения стран"""
    if not selected_countries or len(selected_countries) < 2:
        st.warning("Выберите минимум 2 страны для радарной диаграммы")
        return None

    filtered_df = df[df['Country'].isin(selected_countries)]

    # Нормализация данных для радарной диаграммы
    indicators = ['Life expectancy', 'GDP', 'Urban_population', 'Physicians per thousand',
                  'Gross Primary Education Enrollment (%)', 'Tax revenue (%)']

    fig = go.Figure()

    for country in selected_countries[:5]:  # Максимум 5 стран для читаемости
        country_data = filtered_df[filtered_df['Country'] == country]
        if not country_data.empty:
            values = []
            for indicator in indicators:
                val = country_data[indicator].iloc[0]
                # Нормализация к шкале 0-100
                if indicator == 'GDP':
                    val = min(val / 1e12 * 100, 100)
                elif indicator == 'Life expectancy':
                    val = val / 100 * 100
                values.append(val)

            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=indicators,
                fill='toself',
                name=country,
                line=dict(width=2)
            ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=True,
        title="🎯 Многомерное сравнение стран",
        title_font=dict(size=20, color='#2c3e50'),
        height=600
    )

    return fig


def create_economic_analysis(df, selected_countries):
    """Экономический анализ"""
    filtered_df = df[df['Country'].isin(selected_countries)] if selected_countries else df

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('ВВП по странам', 'Уровень безработицы', 'Налоговые доходы', 'CO2 эмиссии'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )

    top_countries = filtered_df.nlargest(10, 'GDP')

    # ВВП
    fig.add_trace(
        go.Bar(x=top_countries['Country'], y=top_countries['GDP'] / 1e9,
               name='ВВП (млрд $)', marker_color='lightblue'),
        row=1, col=1
    )

    # Безработица
    fig.add_trace(
        go.Scatter(x=filtered_df['Country'], y=filtered_df['Unemployment rate'],  # Изменено
                   mode='markers+lines', name='Безработица (%)', marker_color='red'),
        row=1, col=2
    )

    # Налоговые доходы
    fig.add_trace(
        go.Bar(x=filtered_df['Country'], y=filtered_df['Tax revenue (%)'],  # Изменено
               name='Налоговые доходы (%)', marker_color='green'),
        row=2, col=1
    )
    # create_population_pyramid (эта строка тут не нужна, она просто вызов функции без результата)

    # CO2 эмиссии
    fig.add_trace(
        go.Scatter(x=filtered_df['Population'] / 1e6, y=filtered_df['Co2-Emissions'] / 1e6,
                   mode='markers', name='CO2 (млн тонн)', marker_color='orange',
                   text=filtered_df['Country']),
        row=2, col=2
    )

    fig.update_layout(
        height=800,
        title_text="📊 Комплексный экономический анализ",
        title_font=dict(size=20, color='#2c3e50'),
        showlegend=False
    )

    return fig


def create_demographic_analysis(df, selected_countries):
    """Создание демографического анализа"""
    try:
        filtered_df = df[df['Country'].isin(selected_countries)] if selected_countries else df

        # Удаляем строки с NaN в ключевых демографических показателях
        demo_cols = ['Birth Rate', 'Life expectancy', 'Fertility Rate', 'Infant Mortality']
        available_cols = [col for col in demo_cols if col in filtered_df.columns]

        if not available_cols:
            return None

        clean_df = filtered_df.dropna(subset=available_cols, how='all')

        if clean_df.empty:
            return None

        # Создаем subplot с несколькими графиками
        from plotly.subplots import make_subplots

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Рождаемость vs Продолжительность жизни',
                            'Коэффициент фертильности vs Детская смертность',
                            'Плотность населения', 'Городское население'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )

        # График 1: Birth Rate vs Life Expectancy
        if 'Birth Rate' in clean_df.columns and 'Life expectancy' in clean_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=clean_df['Birth Rate'],
                    y=clean_df['Life expectancy'],
                    mode='markers',
                    text=clean_df['Country'],
                    name='Страны',
                    marker=dict(size=8)
                ),
                row=1, col=1
            )

        # График 2: Fertility Rate vs Infant Mortality
        if 'Fertility Rate' in clean_df.columns and 'Infant Mortality' in clean_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=clean_df['Fertility Rate'],
                    y=clean_df['Infant Mortality'],
                    mode='markers',
                    text=clean_df['Country'],
                    name='Страны',
                    marker=dict(size=8)
                ),
                row=1, col=2
            )

        # График 3: Population Density
        if 'Density (P/Km2)' in clean_df.columns:
            top_density = clean_df.nlargest(10, 'Density (P/Km2)')
            fig.add_trace(
                go.Bar(
                    x=top_density['Country'],
                    y=top_density['Density (P/Km2)'],
                    name='Плотность населения'
                ),
                row=2, col=1
            )

        # График 4: Urban Population
        if 'Urban_population' in clean_df.columns:
            fig.add_trace(
                go.Histogram(
                    x=clean_df['Urban_population'],
                    nbinsx=20,
                    name='Городское население (%)'
                ),
                row=2, col=2
            )

        fig.update_layout(
            title_text="📊 Демографический анализ",
            title_font=dict(size=20, color='#2c3e50'),
            height=800,
            showlegend=False
        )

        return fig
    except Exception as e:
        st.error(f"Ошибка при создании демографического анализа: {str(e)}")
        return None


def create_ecological_analysis(df, selected_countries):
    """Создание экологического анализа"""
    try:
        filtered_df = df[df['Country'].isin(selected_countries)] if selected_countries else df

        # Экологические показатели
        eco_cols = ['Co2-Emissions', 'Forested Area (%)', 'Agricultural Land (%)']
        available_cols = [col for col in eco_cols if col in filtered_df.columns]

        if not available_cols:
            return None

        clean_df = filtered_df.dropna(subset=available_cols, how='all')

        if clean_df.empty:
            return None

        from plotly.subplots import make_subplots

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('CO2 выбросы vs ВВП',
                            'Лесные территории (%)',
                            'Сельскохозяйственные земли (%)',
                            'Экологический баланс'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )

        # График 1: CO2 vs GDP
        if 'Co2-Emissions' in clean_df.columns and 'GDP' in clean_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=clean_df['GDP'],
                    y=clean_df['Co2-Emissions'],
                    mode='markers',
                    text=clean_df['Country'],
                    name='CO2 vs ВВП',
                    marker=dict(size=8, color='red', opacity=0.6)
                ),
                row=1, col=1
            )

        # График 2: Forest Area
        if 'Forested Area (%)' in clean_df.columns:
            top_forest = clean_df.nlargest(15, 'Forested Area (%)')
            fig.add_trace(
                go.Bar(
                    x=top_forest['Country'],
                    y=top_forest['Forested Area (%)'],
                    name='Лесные территории',
                    marker_color='green'
                ),
                row=1, col=2
            )

        # График 3: Agricultural Land
        if 'Agricultural Land (%)' in clean_df.columns:
            top_agri = clean_df.nlargest(15, 'Agricultural Land (%)')
            fig.add_trace(
                go.Bar(
                    x=top_agri['Country'],
                    y=top_agri['Agricultural Land (%)'],
                    name='Сельхоз земли',
                    marker_color='orange'
                ),
                row=2, col=1
            )

        # График 4: Eco Balance (Forest - Agricultural)
        if 'Forested Area (%)' in clean_df.columns and 'Agricultural Land (%)' in clean_df.columns:
            clean_df['Eco_Balance'] = clean_df['Forested Area (%)'] - clean_df['Agricultural Land (%)']
            fig.add_trace(
                go.Scatter(
                    x=clean_df['Forested Area (%)'],
                    y=clean_df['Agricultural Land (%)'],
                    mode='markers',
                    text=clean_df['Country'],
                    name='Экобаланс',
                    marker=dict(
                        size=8,
                        color=clean_df['Eco_Balance'],
                        colorscale='RdYlGn',
                        showscale=True
                    )
                ),
                row=2, col=2
            )

        fig.update_layout(
            title_text="🌍 Экологический анализ",
            title_font=dict(size=20, color='#2c3e50'),
            height=800,
            showlegend=False
        )

        # Поворачиваем подписи на графиках с барами
        fig.update_xaxes(tickangle=45, row=1, col=2)
        fig.update_xaxes(tickangle=45, row=2, col=1)

        return fig
    except Exception as e:
        st.error(f"Ошибка при создании экологического анализа: {str(e)}")
        return None


def create_demographic_metrics(df, selected_countries):
    """Создание демографических метрик"""
    try:
        filtered_df = df[df['Country'].isin(selected_countries)] if selected_countries else df

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            avg_birth = filtered_df['Birth Rate'].dropna().mean() if 'Birth Rate' in filtered_df.columns else 0
            st.markdown("""
            <div class="metric-card">
                <h3>👶 Средняя рождаемость</h3>
                <h2>{:.1f}</h2>
                <p>на 1000 чел.</p>
            </div>
            """.format(avg_birth), unsafe_allow_html=True)

        with col2:
            avg_fertility = filtered_df[
                'Fertility Rate'].dropna().mean() if 'Fertility Rate' in filtered_df.columns else 0
            st.markdown("""
            <div class="metric-card">
                <h3>👪 Коэф. фертильности</h3>
                <h2>{:.2f}</h2>
                <p>детей на женщину</p>
            </div>
            """.format(avg_fertility), unsafe_allow_html=True)

        with col3:
            avg_infant = filtered_df[
                'Infant Mortality'].dropna().mean() if 'Infant Mortality' in filtered_df.columns else 0
            st.markdown("""
            <div class="metric-card">
                <h3>👼 Детская смертность</h3>
                <h2>{:.1f}</h2>
                <p>на 1000 рождений</p>
            </div>
            """.format(avg_infant), unsafe_allow_html=True)

        with col4:
            avg_urban = (
                filtered_df['Urban_population'].dropna().mean()
                if 'Urban_population' in filtered_df.columns else 0
            )
            st.markdown("""
            <div class="metric-card">
                <h3>🏙️ Городское население</h3>
                <h2>{:,.0f}</h2>
                <p>человек в среднем</p>
            </div>
            """.format(avg_urban).replace(",", " "), unsafe_allow_html=True)


    except Exception as e:
        st.error(f"Ошибка при создании демографических метрик: {str(e)}")


def create_ecological_metrics(df, selected_countries):
    """Создание экологических метрик"""
    try:
        filtered_df = df[df['Country'].isin(selected_countries)] if selected_countries else df

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            avg_co2 = filtered_df['Co2-Emissions'].dropna().mean() if 'Co2-Emissions' in filtered_df.columns else 0

            def format_co2_metric(value):
                if value >= 1e6:
                    return f"{value / 1e6:.2f}M"
                elif value >= 1e3:
                    return f"{value / 1e3:.2f}K"
                else:
                    return f"{value:.2f}"

            formatted_avg_co2 = format_co2_metric(avg_co2)

            st.markdown(f"""
            <div class="metric-card">
                <h3>🏭 Средние выбросы CO₂</h3>
                <h2>{formatted_avg_co2}</h2>
                <p>тонн</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            avg_forest = filtered_df[
                'Forested Area (%)'].dropna().mean() if 'Forested Area (%)' in filtered_df.columns else 0
            st.markdown("""
            <div class="metric-card">
                <h3>🌲 Лесные территории</h3>
                <h2>{:.1f}%</h2>
                <p>от общей площади</p>
            </div>
            """.format(avg_forest), unsafe_allow_html=True)

        with col3:
            avg_agri = filtered_df[
                'Agricultural Land (%)'].dropna().mean() if 'Agricultural Land (%)' in filtered_df.columns else 0
            st.markdown("""
            <div class="metric-card">
                <h3>🚜 Сельхоз земли</h3>
                <h2>{:.1f}%</h2>
                <p>от общей площади</p>
            </div>
            """.format(avg_agri), unsafe_allow_html=True)

        with col4:
            if 'Forested Area (%)' in filtered_df.columns and 'Agricultural Land (%)' in filtered_df.columns:
                eco_balance = (filtered_df['Forested Area (%)'].dropna().mean() -
                               filtered_df['Agricultural Land (%)'].dropna().mean())
            else:
                eco_balance = 0
            st.markdown("""
            <div class="metric-card">
                <h3>⚖️ Экобаланс</h3>
                <h2>{:.1f}</h2>
                <p>лес - сельхоз</p>
            </div>
            """.format(eco_balance), unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Ошибка при создании экологических метрик: {str(e)}")


def main():
    # Заголовок приложения
    st.markdown("""
    <div style='text-align: center; padding: 2rem 0;'>
        <h1 style='color: #2c3e50; font-size: 3rem; margin-bottom: 0;'>🌍 Глобальный Аналитический Дашборд</h1>
        <h3 style='color: #7f8c8d; margin-top: 0;'>Исследование данных стран мира 2023</h3>
    </div>
    """, unsafe_allow_html=True)

    # Загрузка данных
    df_original = load_data()  # Сохраняем оригинальные данные
    df = df_original.copy()  # Создаем копию для фильтрации

    # Боковая панель
    st.sidebar.markdown("## 🎛️ Панель управления")

    # Фильтры
    st.sidebar.markdown("### 🔍 Фильтры")

    # Выбор стран (всегда используем оригинальные данные)
    all_countries = df_original['Country'].unique().tolist()
    selected_countries = st.sidebar.multiselect(
        "Выберите страны:",
        options=all_countries,
        default=[],
        help="Оставьте пустым для анализа всех стран"
    )

    # Диапазон населения
    apply_population_filter = st.sidebar.checkbox(
        "Применить фильтр по населению",
        value=False,
        help="Включите для фильтрации по численности населения"
    )

    if apply_population_filter and 'Population' in df.columns and df['Population'].notna().any():
        # Дополнительная проверка на NaN и Infinity, чтобы избежать ошибок с min/max
        valid_population = df_original['Population'].dropna()
        if not valid_population.empty:
            pop_min_val = int(valid_population.min() / 1e6) if valid_population.min() > 0 else 0
            pop_max_val = int(valid_population.max() / 1e6) if valid_population.max() > 0 else 100

            # Убеждаемся, что min < max
            if pop_min_val >= pop_max_val:
                pop_max_val = pop_min_val + 100

            pop_range = st.sidebar.slider(
                "Диапазон населения (млн):",
                min_value=pop_min_val,
                max_value=pop_max_val,
                value=(pop_min_val, pop_max_val),
                help="Фильтр по численности населения"
            )

            # Применение фильтра по населению ТОЛЬКО если не выбраны конкретные страны
            if not selected_countries:
                df = df[(df['Population'] >= pop_range[0] * 1e6) & (df['Population'] <= pop_range[1] * 1e6)]
            else:
                # Если выбраны конкретные страны, фильтр по населению применяется только к ним
                selected_df = df_original[df_original['Country'].isin(selected_countries)]
                other_df = df_original[~df_original['Country'].isin(selected_countries)]
                other_df = other_df[
                    (other_df['Population'] >= pop_range[0] * 1e6) & (other_df['Population'] <= pop_range[1] * 1e6)]
                df = pd.concat([selected_df, other_df])

                st.sidebar.info(f"ℹ️ Выбранные страны ({len(selected_countries)}) всегда включены в анализ")

    # Показать информацию о текущих данных
    if selected_countries:
        # Проверяем, какие из выбранных стран действительно есть в данных
        available_countries = [country for country in selected_countries if country in df['Country'].values]
        missing_countries = [country for country in selected_countries if country not in df['Country'].values]

        if missing_countries:
            st.sidebar.warning(f"⚠️ Страны не найдены в данных: {', '.join(missing_countries)}")

        if available_countries:
            st.sidebar.success(
                f"✅ Анализ по {len(available_countries)} странам: {', '.join(available_countries[:3])}{'...' if len(available_countries) > 3 else ''}")
        else:
            st.sidebar.error("❌ Ни одной выбранной страны не найдено в данных!")
            # Сбрасываем выбор, если ни одной страны не найдено
            selected_countries = []

    # Выбор режима анализа
    analysis_mode = st.sidebar.selectbox(
        "🎯 Режим анализа:",
        ["Обзор", "Экономический", "Демографический", "Экологический", "Сравнительный"]
    )

    # Отладочная информация (можно убрать в продакшене)
    if st.sidebar.checkbox("🔧 Показать отладочную информацию", value=False):
        st.sidebar.write(f"Всего стран в данных: {len(df)}")
        st.sidebar.write(f"Выбрано стран: {len(selected_countries) if selected_countries else 0}")
        if selected_countries:
            st.sidebar.write("Выбранные страны:", selected_countries[:5])

    # Основной контент
    if analysis_mode == "Обзор":
        st.markdown("## 📊 Общий обзор")
        # Карточки с метриками
        create_metric_cards(df, selected_countries)
        # Два столбца для графиков
        col1, col2 = st.columns(2)
        with col1:
            fig_scatter = create_gdp_vs_life_expectancy(df, selected_countries)
            if fig_scatter:
                st.plotly_chart(fig_scatter, use_container_width=True)
            else:
                st.info("📊 График ВВП vs Ожидаемая продолжительность жизни недоступен")
        with col2:
            fig_pop = create_population_pyramid(df, selected_countries)
            if fig_pop:
                st.plotly_chart(fig_pop, use_container_width=True)
            else:
                st.info("📊 График населения недоступен")
        # Тепловая карта корреляций
        fig_corr = create_correlation_heatmap(df, selected_countries)
        if fig_corr:
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info("📊 Тепловая карта корреляций недоступна")
        # Интерактивная таблица (всегда показываем)
        st.markdown("## 📋 Интерактивная таблица данных")

        # Фильтр для таблицы
        display_df = df[df['Country'].isin(selected_countries)] if selected_countries else df

        # Поиск по таблице
        search_term = st.text_input("🔎 Поиск по странам:", "")
        if search_term:
            display_df = display_df[display_df['Country'].str.contains(search_term, case=False, na=False)]

        if not display_df.empty:
            # Фильтр колонок
            st.markdown("### 🎛️ Настройка отображаемых колонок")

            # Создаем три колонки для управления фильтром
            col_filter1, col_filter2, col_filter3 = st.columns([2, 1, 1])

            with col_filter1:
                st.markdown("**Выберите колонки для отображения:**")

            with col_filter2:
                if st.button("✅ Выбрать все", key="select_all_cols"):
                    st.session_state.selected_columns = display_df.columns.tolist()

            with col_filter3:
                if st.button("❌ Снять все", key="deselect_all_cols"):
                    st.session_state.selected_columns = ['Country']  # Оставляем только Country

            # Инициализация состояния, если его нет
            if 'selected_columns' not in st.session_state:
                st.session_state.selected_columns = display_df.columns.tolist()

            if 'Country' not in st.session_state.selected_columns:
                st.session_state.selected_columns.insert(0, 'Country')

            # Группировка колонок по категориям для лучшей организации
            column_categories = {
                "🏛️ Основная информация": ['Country', 'Abbreviation', 'Capital/Major City', 'Largest City',
                                           'Official Language'],
                "👥 Демография": ['Population', 'Density (P/Km2)', 'Birth Rate', 'Fertility Rate', 'Life expectancy',
                                 'Infant Mortality', 'Maternal Mortality Ratio', 'Urban_population'],
                "💰 Экономика": ['GDP', 'CPI', 'CPI Change (%)', 'Currency_Code', 'Minimum Wage',
                                'Tax revenue (%)', 'Total tax rate', 'Unemployment rate',
                                'Population: Labor force participation (%)'],
                "🎓 Образование": ['Gross Primary Education Enrollment (%)',
                                  'Gross Tertiary Education Enrollment (%)'],
                "🏥 Здравоохранение": ['Physicians per thousand', 'Out of pocket health expenditure'],
                "🌍 География и экология": ['Land Area (Km2)', 'Agricultural Land (%)', 'Forested Area (%)',
                                           'Co2-Emissions', 'Latitude', 'Longitude'],
                "🛡️ Прочее": ['Armed Forces Size', 'Calling Code', 'Gasoline_Price']
            }

            # Создаем expandable секции для каждой категории
            available_columns = display_df.columns.tolist()

            for category, cols in column_categories.items():
                # Фильтруем только существующие колонки
                existing_cols = [col for col in cols if col in available_columns and col != 'Country']

                if existing_cols:  # Показываем категорию только если в ней есть колонки
                    with st.expander(f"{category} ({len(existing_cols)} колонок)", expanded=False):
                        # Создаем чекбоксы в несколько колонок для компактности
                        checkbox_cols = st.columns(min(3, len(existing_cols)))

                        for idx, col in enumerate(existing_cols):
                            with checkbox_cols[idx % len(checkbox_cols)]:
                                current_state = col in st.session_state.selected_columns
                                new_state = st.checkbox(
                                    col.replace('_', ' ').title(),
                                    value=current_state,
                                    key=f"col_filter_{col}"
                                )

                                # Обновляем состояние
                                if new_state and col not in st.session_state.selected_columns:
                                    st.session_state.selected_columns.append(col)
                                elif not new_state and col in st.session_state.selected_columns:
                                    st.session_state.selected_columns.remove(col)

            # Показываем информацию о выбранных колонках
            selected_count = len(st.session_state.selected_columns)
            total_count = len(available_columns)

            if selected_count > 0:
                st.markdown(f"""
                <div class="highlight">
                    📊 <strong>Выбрано колонок:</strong> {selected_count} из {total_count}
                </div>
                """, unsafe_allow_html=True)

                # Фильтруем DataFrame по выбранным колонкам
                st.session_state.selected_columns = [
                    col for col in st.session_state.selected_columns if col in display_df.columns
                ]
                display_df_filtered = display_df[st.session_state.selected_columns]

                # Отображение таблицы с дополнительными опциями
                st.markdown("### 📋 Таблица данных:")

                # Опции отображения
                display_options_col1, display_options_col2 = st.columns(2)

                with display_options_col2:
                    table_height = 400

                # Отображаем таблицу
                st.dataframe(
                    display_df_filtered,
                    use_container_width=True,
                    height=table_height,
                )

                # Дополнительная информация
                st.markdown(f"""
                <div style='color: #666; font-size: 0.9em; margin-top: 1rem;'>
                    📈 Отображено строк: <strong>{len(display_df_filtered)}</strong> | 
                    📊 Колонок: <strong>{len(st.session_state.selected_columns)}</strong>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)

                def to_excel(df):
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        df.to_excel(writer, index=False, sheet_name='Data')
                    processed_data = output.getvalue()
                    return processed_data

                excel_data = to_excel(display_df_filtered)

                st.download_button(
                    label="## 📥 Скачать таблицу в Excel",
                    data=excel_data,
                    file_name="filtered_data.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

            else:
                st.warning("⚠️ Выберите хотя бы одну колонку для отображения таблицы")

        else:
            st.warning("Нет данных для отображения с текущими фильтрами")

    elif analysis_mode == "Экономический":
        st.markdown("## 💰 Экономический анализ")
        fig_econ = create_economic_analysis(df, selected_countries)
        if fig_econ:
            st.plotly_chart(fig_econ, use_container_width=True)
        else:
            st.info("📊 Экономический анализ недоступен")
        # Топ стран по ВВП
        st.markdown("### 🏆 Топ-10 стран по ВВП")
        filtered_df = df[df['Country'].isin(selected_countries)] if selected_countries else df
        if safe_column_check(filtered_df, 'Unemployment rate'):
            top_gdp = filtered_df.nlargest(10, 'GDP')[['Country', 'GDP', 'Population', 'Unemployment rate']]
            st.dataframe(top_gdp, use_container_width=True)
        else:
            st.info("Данные об уровне безработицы недоступны для вывода.")

    elif analysis_mode == "Демографический":
        st.markdown("## 👥 Демографический анализ")

        # Демографические метрики
        create_demographic_metrics(df, selected_countries)

        # Демографические графики
        fig_demo = create_demographic_analysis(df, selected_countries)
        if fig_demo:
            st.plotly_chart(fig_demo, use_container_width=True)
        else:
            st.info("📊 Демографический анализ недоступен")

        # Таблица демографических показателей
        st.markdown("### 📋 Демографические показатели")
        filtered_df = df[df['Country'].isin(selected_countries)] if selected_countries else df
        demo_columns = ['Country', 'Population', 'Density (P/Km2)', 'Birth Rate',
                        'Life expectancy', 'Fertility Rate', 'Infant Mortality', 'Urban_population']
        available_demo_cols = [col for col in demo_columns if col in filtered_df.columns]

        if available_demo_cols:
            demo_table = filtered_df[available_demo_cols].dropna(how='all')
            if not demo_table.empty:
                st.dataframe(demo_table.sort_values('Population', ascending=False), use_container_width=True)
            else:
                st.warning("Нет демографических данных для отображения")
        else:
            st.warning("Демографические колонки не найдены")

    elif analysis_mode == "Экологический":
        st.markdown("## 🌍 Экологический анализ")

        # Экологические метрики
        create_ecological_metrics(df, selected_countries)

        # Экологические графики
        fig_eco = create_ecological_analysis(df, selected_countries)
        if fig_eco:
            st.plotly_chart(fig_eco, use_container_width=True)
        else:
            st.info("📊 Экологический анализ недоступен")

        # Таблица экологических показателей
        st.markdown("### 📋 Экологические показатели")
        filtered_df = df[df['Country'].isin(selected_countries)] if selected_countries else df
        eco_columns = ['Country', 'Co2-Emissions', 'Forested Area (%)', 'Agricultural Land (%)',
                       'Land Area (Km2)', 'Population', 'GDP']
        available_eco_cols = [col for col in eco_columns if col in filtered_df.columns]

        if available_eco_cols:
            eco_table = filtered_df[available_eco_cols].dropna(how='all')
            if not eco_table.empty:
                # Добавляем расчетные показатели
                if 'Co2-Emissions' in eco_table.columns and 'Population' in eco_table.columns:
                    eco_table['CO2 на душу населения'] = eco_table['Co2-Emissions'] / eco_table['Population']
                if 'Co2-Emissions' in eco_table.columns and 'GDP' in eco_table.columns:
                    eco_table['CO2/GDP интенсивность'] = eco_table['Co2-Emissions'] / eco_table['GDP']

                # Форматируем выбросы CO2
                def format_co2(value):
                    if value >= 1e6:
                        return f"{value / 1e6:.2f}M"
                    elif value >= 1e3:
                        return f"{value / 1e3:.2f}K"
                    else:
                        return f"{value:.2f}"

                eco_table['Форматированные CO2'] = eco_table['Co2-Emissions'].apply(format_co2)

                # Упорядочиваем колонки
                display_cols = ['Country', 'Форматированные CO2', 'Forested Area (%)',
                                'Agricultural Land (%)', 'Land Area (Km2)', 'Population',
                                'GDP', 'CO2 на душу населения', 'CO2/GDP интенсивность']
                display_cols = [col for col in display_cols if col in eco_table.columns]

                # Сортируем по исходным числовым значениям выбросов
                eco_table = eco_table.sort_values('Co2-Emissions', ascending=False)

                # Показываем таблицу
                st.dataframe(eco_table[display_cols], use_container_width=True)

            else:
                st.warning("Нет экологических данных для отображения")

        else:
            st.warning("Экологические колонки не найдены")

    elif analysis_mode == "Сравнительный":
        st.markdown("## ⚖️ Сравнительный анализ")
        if len(selected_countries) >= 2:
            fig_radar = create_multi_indicator_radar(df, selected_countries)
            if fig_radar:
                st.plotly_chart(fig_radar, use_container_width=True)
            else:
                st.info("📊 Радарная диаграмма недоступна. Проверьте данные или выбранные страны.")
        else:
            st.info("Выберите минимум 2 страны для сравнительного анализа")
        # Таблица сравнения
        if selected_countries:
            st.markdown("### 📋 Детальное сравнение выбранных стран")
            comparison_df = df[df['Country'].isin(selected_countries)]
            if not comparison_df.empty:
                st.dataframe(comparison_df, use_container_width=True)
            else:
                st.warning("Выбранные страны не найдены в отфильтрованных данных")

    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #7f8c8d; padding: 2rem 0;'>
        <p>📊 Глобальный аналитический дашборд стран мира • Создано с использованием Streamlit и Plotly</p>
        <p>🔄 Данные обновляются в реальном времени при изменении фильтров</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
