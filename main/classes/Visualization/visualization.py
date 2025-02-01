import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from bokeh.plotting import figure, show, output_notebook
from bokeh.layouts import gridplot
from bokeh.palettes import Category20
import altair as alt
from scipy.stats import kurtosis, skew
import numpy as np

class Visualization:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def plot_time_series(self, stocks, start_date=None, end_date=None):
        filtered_data = self.data.copy()
        filtered_data['Date'] = pd.to_datetime(filtered_data['Date'])
        if start_date and end_date:
            filtered_data = filtered_data[(filtered_data['Date'] >= start_date) & 
                                          (filtered_data['Date'] <= end_date)]
        plt.figure(figsize=(12, 6))
        for stock in stocks:
            plt.plot(filtered_data['Date'], filtered_data[stock], label=stock)
        plt.xlabel('Date')
        plt.ylabel('Closing Price')
        plt.title('Time Series of Selected Stocks')
        plt.legend()
        plt.grid()
        plt.show()

    def plot_bollinger_bands(self, stock, window=20, start_date=None, end_date=None):
        data_frame = self.data.copy()
        data_frame['Date'] = pd.to_datetime(data_frame['Date'])
        
        data_frame['SMA'] = data_frame[stock].rolling(window=window).mean()
        data_frame['Banda Superior'] = data_frame['SMA'] + (2 * data_frame[stock].rolling(window=window).std())
        data_frame['Banda Inferior'] = data_frame['SMA'] - (2 * data_frame[stock].rolling(window=window).std())
        
        if start_date and end_date:
            filtered_df = data_frame[(data_frame['Date'] >= start_date) & (data_frame['Date'] <= end_date)]
        else:
            filtered_df = data_frame
        
        plt.figure(figsize=(12, 6))
        plt.plot(filtered_df['Date'], filtered_df[stock], label=stock, color='blue', alpha=0.5)
        plt.plot(filtered_df['Date'], filtered_df['SMA'], label='Média Móvel (SMA)', color='orange', linestyle='--')
        plt.plot(filtered_df['Date'], filtered_df['Banda Superior'], label='Banda Superior', color='green', linestyle='--')
        plt.plot(filtered_df['Date'], filtered_df['Banda Inferior'], label='Banda Inferior', color='red', linestyle='--')
        plt.fill_between(filtered_df['Date'], filtered_df['Banda Superior'], filtered_df['Banda Inferior'], color='gray', alpha=0.2)
        plt.title(f'Bandas de Bollinger - {stock} (Período Filtrado)')
        plt.xlabel('Data')
        plt.ylabel('Preço')
        plt.legend()
        plt.grid()
        plt.show()

    def plot_returns(self, stock, start_date=None, end_date=None):
        data_frame = self.data.copy()
        data_frame['Date'] = pd.to_datetime(data_frame['Date'])
        
        if start_date and end_date:
            filtered_data = data_frame[(data_frame['Date'] >= start_date) & 
                                       (data_frame['Date'] <= end_date)].copy()
        else:
            filtered_data = data_frame.copy()
        
        filtered_data.loc[:, 'Returns'] = filtered_data[stock].pct_change() * 100
        plt.figure(figsize=(12, 6))
        plt.plot(filtered_data['Date'], filtered_data['Returns'], label=f'{stock} Returns')
        plt.axhline(0, color='red', linestyle='--', linewidth=0.8)
        plt.xlabel('Date')
        plt.ylabel('Percentage Return')
        plt.title(f'Returns for {stock}')
        plt.legend()
        plt.grid()
        plt.show()

    def plot_histograms(self, stocks):
        sns.set(style="whitegrid")
        sns.set_palette("husl")
        plt.figure(figsize=(20, 10))
        for i, stock in enumerate(stocks, 1):
            plt.subplot(3, 4, i)
            sns.histplot(self.data[stock], bins=30, kde=True, color='blue', edgecolor='black', alpha=0.7)
            plt.title(f'{stock}', fontsize=14, fontweight='bold')
            plt.xlabel('Price', fontsize=12)
            plt.ylabel('Frequency', fontsize=12)
        plt.tight_layout()
        plt.show()

    def plot_average_prices(self, data=None):
        if data is None:
            data = self.data
        avg_prices = data.iloc[:, 1:].mean()  # Calcula a média das ações (exclui a coluna de data)
        
        plt.figure(figsize=(12, 6))
        avg_prices.sort_values().plot(kind='bar', color='skyblue')
        plt.title('Average Closing Prices by Stock')
        plt.ylabel('Average Price')
        plt.xlabel('Stock')
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()

    def plot_boxplot(self, stocks, title):
        plt.figure(figsize=(12, 6))
        self.data[stocks].boxplot()
        plt.title(title)
        plt.ylabel('Price')
        plt.xlabel('Stock')
        plt.xticks(rotation=45)
        plt.grid(alpha=0.5)
        plt.show()

    def plot_correlation_matrix(self, data=None, title='Correlation Between Stocks'):
        if data is None:
            data = self.data
        # Excluir a coluna 'Date' se existir
        if 'Date' in data.columns:
            data = data.drop(columns=['Date'])
        correlation = data.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f", cbar=True)
        plt.title(title)
        plt.show()

    def plot_chart_bokeh(self, ls_stock):
        output_notebook()
        num_stock = len(ls_stock)
        palette = Category20[num_stock]
        p = figure(x_axis_type='datetime', title='Preços de Fechamento - Todas as Ações',
                   width=1200, height=600)
        for stock, color in zip(ls_stock, palette):
            p.line(self.data['Date'], self.data[stock], legend_label=stock, line_width=2, color=color)
        p.legend.title = "Ações"
        p.legend.location = "top_left"
        p.legend.click_policy = "hide"
        show(p)

    def plot_chart_altair(self, stock):
        line_original = alt.Chart(self.data).mark_line().encode(
            x=alt.X('Date:T', title='Data'),
            y=alt.Y(f'{stock}:Q', title='Fechamento'),
            tooltip=[alt.Tooltip('Date:T', title='Data'), alt.Tooltip(f'{stock}:Q', title='Preço')]
        ).properties(
            title=f'Tendência de Fechamento - {stock}',
            width=1200,
            height=600
        )
        line_tendence = alt.Chart(self.data).transform_regression(
            'Date', stock, method='poly', order=5
        ).mark_line(color='red', strokeDash=[5, 5]).encode(
            x='Date:T',
            y=f'{stock}:Q'
        )
        chart = (line_original + line_tendence).interactive()
        chart.show()

    def plot_kurtosis_skewness(self, ls_stock):
        output_notebook()
        plots = []
        for stock in ls_stock:
            data = self.data[["Date", stock]].dropna()
            stock_skewness = skew(data[stock])
            stock_kurtosis = kurtosis(data[stock])
            mean, std = data[stock].mean(), data[stock].std()
            x = np.linspace(data[stock].min(), data[stock].max(), 1000)
            normal_pdf = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std)**2)
            p = figure(title=f"Distribution of {stock} \n Skewness: {stock_skewness:.2f}, Kurtosis: {stock_kurtosis:.2f}",
                       x_axis_label='Price', y_axis_label='Frequency',
                       width=600, height=400)
            hist, edges = np.histogram(data[stock], bins=50, density=True)
            p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], fill_color="blue", line_color="black", alpha=0.7)
            p.line(x, normal_pdf, line_width=2, color="red", legend_label="Normal Distribution")
            p.legend.location = "top_right"
            p.legend.title = "Legenda"
            p.grid.grid_line_alpha = 0.3
            plots.append(p)
        grid = gridplot([plots[i:i+2] for i in range(0, len(plots), 2)])
        show(grid)

    def plot_violin_boxplot(self, data, title, ylabel, extra_data=None, extra_title=None):
        if extra_data is not None:
            fig, axes = plt.subplots(1, 2, figsize=(18, 8), sharey=True)
            sns.violinplot(data=data, inner=None, linewidth=1, ax=axes[0])
            sns.boxplot(data=data, whis=1.5, width=0.2, linewidth=1, color='lightgrey', ax=axes[0])
            axes[0].set_title(title, fontsize=16)
            axes[0].set_ylabel(ylabel, fontsize=12)
            axes[0].set_xlabel('Stocks', fontsize=12)
            axes[0].tick_params(axis='x', rotation=45)
            axes[0].grid(axis='y', linestyle='--', alpha=0.7)
            sns.violinplot(data=extra_data, inner=None, linewidth=1, ax=axes[1])
            sns.boxplot(data=extra_data, whis=1.5, width=0.2, linewidth=1, color='lightgrey', ax=axes[1])
            axes[1].set_title(extra_title if extra_title else f'{title} (Dataset 2)', fontsize=16)
            axes[1].set_xlabel('Stocks', fontsize=12)
            axes[1].tick_params(axis='x', rotation=45)
            axes[1].grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
        else:
            plt.figure(figsize=(12, 8))
            sns.violinplot(data=data, inner=None, linewidth=1)
            sns.boxplot(data=data, whis=1.5, width=0.2, linewidth=1, color='lightgrey')
            plt.title(title, fontsize=16)
            plt.ylabel(ylabel, fontsize=12)
            plt.xlabel('Stocks', fontsize=12)
            plt.xticks(rotation=45)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
        plt.show()