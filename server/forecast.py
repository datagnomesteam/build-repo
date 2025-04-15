
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import plotly.graph_objects as go
import pandas as pd
import streamlit as st

def build_forecast_data(df, date_field, freq):
    df[date_field] = pd.to_datetime(df[date_field], errors='coerce')
    df = df[df[date_field] >= pd.Timestamp('1902-01-01')]
    counts = df.groupby(pd.Grouper(key=date_field, freq=freq), dropna=True).size().reset_index(name='count')
    counts = counts.sort_values(date_field)
    counts.set_index(date_field, inplace=True)
    counts.index = pd.to_datetime(counts.index)
    return counts

def forecast(counts_df, date_field, freq):
    # fit the Exponential Smoothing Model
    try:
        model = ExponentialSmoothing(counts_df['count'], trend='add', seasonal='add', seasonal_periods=12)
    except ValueError:
        combined_df = counts_df.reset_index()
        combined_df['type'] = ['Actual'] * len(counts_df)
        return {"df": combined_df, "mse": None, "params": None, "forecasted": False}
    else:
        fit = model.fit()
        params = fit.params
        mse = round(fit.sse / len(counts_df))

        # generate Forecasts
        forecast_periods = 24  # forecasting 24 months ahead
        forecast_index = pd.date_range(start=counts_df.index[-1] + pd.DateOffset(months=1), periods=forecast_periods, freq=freq)
        forecast = fit.forecast(forecast_periods)
        forecast_series = pd.Series(forecast, index=forecast_index)

        # combine actual and forecasted Data
        combined_series = pd.concat([counts_df['count'], forecast_series])
        combined_df = combined_series.reset_index()
        combined_df.columns = [date_field, 'count']
        combined_df['type'] = ['Actual'] * len(counts_df) + ['Forecast'] * forecast_periods
        return {"df": combined_df, "mse": mse, "params": params, "forecasted": True}

def plot_timeseries(df, forecasted, mse, date_field, page):
    policy_changes = [
        {'date': '1906-01-01', 'label': 'Pure Food and Drugs Act'},
        {'date': '1938-01-01', 'label': 'Federal Food, Drug, and Cosmetic Act'},
        {'date': '1944-01-01', 'label': 'Public Health Service Act'},
        {'date': '1968-01-01', 'label': 'Radiation Control for Health and Safety Act'},
        {'date': '1976-01-01', 'label': 'Medical Device Amendments to the FD&C Act'},
        {'date': '1990-01-01', 'label': 'Safe Medical Devices Act'},
        {'date': '1992-01-01', 'label': 'Mammography Quality Standards Act'},
        {'date': '1997-01-01', 'label': 'FDA Modernization Act'},
        {'date': '2002-01-01', 'label': 'Medical Device User Fee and Modernization Act'},
        {'date': '2007-01-01', 'label': 'FDA Amendments Act'},
        {'date': '2012-01-01', 'label': 'FDA Safety and Innovation Act'},
        {'date': '2016-01-01', 'label': '21st Century Cures Act'},
        {'date': '2017-01-01', 'label': 'FDA Reauthorization Act'},
        {'date': '2020-01-01', 'label': 'Coronavirus Aid, Relief, and Economic Security Act'},
        {'date': '2022-01-01', 'label': 'FDAUFRA; FDORA; PREVENT Pandemics Act'}
    ]

    # Filter policy changes to include only those within the data range
    valid_policy_changes = [
        policy for policy in policy_changes
        if df[date_field].min() <= pd.to_datetime(policy['date']) <= df[date_field].max()
    ]
    
    # plot with plotly
    fig = go.Figure()

    # plot actual data
    fig.add_trace(go.Scatter(
        x=df[df['type'] == 'Actual'][date_field],
        y=df[df['type'] == 'Actual']['count'],
        mode='lines',
        name='Actual',
        line=dict(color='blue')
    ))

    if forecasted:
        # plot forecasted data
        fig.add_trace(go.Scatter(
            x=df[df['type'] == 'Forecast'][date_field],
            y=df[df['type'] == 'Forecast']['count'],
            mode='lines',
            name='Forecast',
            line=dict(color='red', dash='dash')
        ))
    
    # add vertical lines for each valid policy change
    for policy in valid_policy_changes:
        fig.add_vline(
            x=pd.to_datetime(policy['date']),
            line_width=2,
            line_dash="solid",
            line_color="white"
        )
        fig.add_annotation(
            x=pd.to_datetime(policy['date']),
            y=df['count'].max(),
            text=policy['label'],
            showarrow=False,
            font=dict(size=10, color="white"),
            xanchor='left',
            textangle=-30,
            yshift=150
        )
    # add footer with mse and hyperlink to medical device policy
    fig.add_annotation(
        x=0,
        xshift=100,
        y=0,
        yshift=-35,
        xref="paper",
        yref="paper",
        text=f"Holt-Winters MSE: {mse} | <a href='https://www.fda.gov/medical-devices/overview-device-regulation/history-medical-device-regulation-oversight-united-states' target='_blank'>Read about Medical Device Policy</a>",
        showarrow=False,
        font=dict(size=12, color="grey"),
        align="center",
        xanchor="center",
        yanchor="top"
    )

    fig.update_layout(
        title=f"{page} Over Time",
        xaxis_title='Date',
        yaxis_title=f"Number of {page}",
        legend=dict(x=0, y=1),
        hovermode='x unified'
    )

    return fig