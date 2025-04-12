import pandas as pd
import matplotlib.pyplot as plt
import sqlalchemy
from configs import db_credentials
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose

# Database connection setup
db_connection_str = f"postgresql+psycopg2://{db_credentials['username']}:{db_credentials['password']}@{db_credentials['host']}:{db_credentials['port']}/{db_credentials['database']}"
db_engine = sqlalchemy.create_engine(db_connection_str)

# Function to load data from SQL view
def load_data(view_name):
    query = f'SELECT * FROM {view_name}'
    return pd.read_sql(query, con=db_engine)

# Load data
integrated_df = load_data('integrated')

# Extract year from date strings
integrated_df['year'] = integrated_df['event_date'].str.extract(r'^(\d{4})').astype(float)
integrated_df = integrated_df[(integrated_df['year'] >= 1677) & (integrated_df['year'] <= 2262)]
integrated_df['event_date'] = pd.to_datetime(integrated_df['event_date'], errors='coerce')

daily_counts = integrated_df.groupby('event_date')[['recall', 'injury', 'malfunction', 'death', 'other']].sum().asfreq('D').fillna(0)

print(len(daily_counts))

model_recall = ExponentialSmoothing(
    daily_counts['recall'],
    trend='add',
    seasonal='add',
    seasonal_periods=7  # Weekly seasonality
)

fit_recall = model_recall.fit()
daily_counts['recalls_smoothed'] = fit_recall.fittedvalues
forecast = fit_recall.forecast(12)

# Plot the original data and fitted values
daily_counts['recall'].plot(label='Original', figsize=(10, 6))
daily_counts['recalls_smoothed'].plot(label='Fitted')

# Plot the forecasted values
forecast.plot(label='Forecast', legend=True)
plt.title('Holt-Winters Forecast')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.show()