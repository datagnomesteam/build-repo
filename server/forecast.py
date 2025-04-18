
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import plotly.graph_objects as go
import pandas as pd
import streamlit as st
import numpy as np
from sklearn.metrics import mean_squared_error

def build_forecast_data(df, date_field, freq):
    df[date_field] = pd.to_datetime(df[date_field], errors='coerce')
    counts = df.groupby(pd.Grouper(key=date_field, freq=freq), dropna=True).size().reset_index(name='count')
    counts = counts.sort_values(date_field)
    counts.set_index(date_field, inplace=True)
    counts.index = pd.to_datetime(counts.index)
    return counts


def forecast(counts_df, date_field, freq):
    holdout_size = round(len(counts_df) * .10) # holdout 6 months for validation
    seasonal_periods = 12 # 12 month intervals
    forecast_periods = holdout_size + 24 # forecast 24 months past held out data
    # fit the Exponential Smoothing Model
    try:
        train = counts_df.iloc[:-holdout_size]
        test = counts_df.iloc[-holdout_size:]
        if (counts_df['count'] < 1).any():
            model = ExponentialSmoothing(train['count'], trend='add', seasonal='add', seasonal_periods=seasonal_periods)
        else:
            model = ExponentialSmoothing(train['count'], trend='add', seasonal='mul', seasonal_periods=seasonal_periods)
    except ValueError as e:
        print(e)
        combined_df = counts_df.reset_index()
        combined_df['type'] = ['Actual'] * len(counts_df)
        return {"df": combined_df, "rmse": None, "params": None, "forecasted": False}
    else:
        # generate held out forecasts for validation
        fit = model.fit()
        heldout_forecast = fit.forecast(steps=holdout_size)
        rmse = round(np.sqrt(mean_squared_error(test['count'], heldout_forecast)))

        # generate forecasts
        forecast_index = pd.date_range(start=train.index[-1] + pd.DateOffset(months=1), periods=forecast_periods, freq=freq)
        forecast = fit.forecast(forecast_periods)
        forecast_series = pd.Series(forecast, index=forecast_index)

        # combine actual and forecasted Data
        actual_df = counts_df.reset_index()
        actual_df['type'] = 'Actual'

        heldout_df = pd.DataFrame({
            date_field: test.index,
            'count': heldout_forecast.values,
            'type': 'Held-Out Forecast'
        })

        forecast_df = pd.DataFrame({
            date_field: forecast_index,
            'count': forecast_series.values,
            'type': 'Forecast'
        })
        
        # Combine all DataFrames
        combined_df = pd.concat([actual_df, heldout_df, forecast_df], ignore_index=True)
        return {"df": combined_df, "rmse": rmse, "forecasted": True}

def plot_timeseries(df, forecasted, rmse, date_field, page):
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
        # plot forecasted data
        fig.add_trace(go.Scatter(
            x=df[df['type'] == 'Held-Out Forecast'][date_field],
            y=df[df['type'] == 'Held-Out Forecast']['count'],
            mode='lines',
            name='Held-Out Forecast',
            line=dict(color='red')
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
        text=f"Held-Out RMSE: {rmse} | <a href='https://www.fda.gov/medical-devices/overview-device-regulation/history-medical-device-regulation-oversight-united-states' target='_blank'>Read about Medical Device Policy</a>",
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
        hovermode='x unified',
        yaxis_range=[0 if df['count'].min() > 0 else df['count'].min() - 100, df['count'].max() + 100]
    )

    return fig