import datetime
from typing import List, Dict
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import io
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

class Developer:
    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role
        self.tasks: List[Dict] = []
        self.hours_worked: Dict[datetime.date, float] = {}
        self.burnout_rate: Dict[datetime.date, float] = {}

class BurnoutPreventionSystem:
    def __init__(self, data_file: str):
        self.developers: List[Developer] = []
        self.burnout_model = LinearRegression()
        self.hours_model = LinearRegression()
        self.workload_model = LinearRegression()
        self.scaler = StandardScaler()
        self.data = pd.read_csv('developer_data.csv')

    def prepare_data(self):
        X = self.data[['weekly_hours', 'weekly_tasks']]
        y_burnout = self.data['burnout_rate']
        y_hours = self.data['next_week_hours']
        y_workload = self.data['next_week_tasks']
        return X, y_burnout, y_hours, y_workload

    def train_models(self):
        X, y_burnout, y_hours, y_workload = self.prepare_data()
        X = self.scaler.fit_transform(X)

        X_train, X_test, y_burnout_train, y_burnout_test = train_test_split(X, y_burnout, test_size=0.2, random_state=42)
        self.burnout_model.fit(X_train, y_burnout_train)
        burnout_pred = self.burnout_model.predict(X_test)
        burnout_mae = mean_absolute_error(y_burnout_test, burnout_pred)
        burnout_mse = mean_squared_error(y_burnout_test, burnout_pred)

        self.hours_model.fit(X_train, y_hours.iloc[y_burnout_train.index])
        hours_pred = self.hours_model.predict(X_test)
        hours_mae = mean_absolute_error(y_hours.iloc[y_burnout_test.index], hours_pred)
        hours_mse = mean_squared_error(y_hours.iloc[y_burnout_test.index], hours_pred)

        self.workload_model.fit(X_train, y_workload.iloc[y_burnout_train.index])
        workload_pred = self.workload_model.predict(X_test)
        workload_mae = mean_absolute_error(y_workload.iloc[y_burnout_test.index], workload_pred)
        workload_mse = mean_squared_error(y_workload.iloc[y_burnout_test.index], workload_pred)

        return (burnout_mae, burnout_mse), (hours_mae, hours_mse), (workload_mae, workload_mse)

    def predict_burnout(self, weekly_hours: float, weekly_tasks: int):
        recent_data = np.array([[weekly_hours, weekly_tasks]])
        recent_data = self.scaler.transform(recent_data)
        predicted_burnout = self.burnout_model.predict(recent_data)[0]
        return max(0, min(1, predicted_burnout))  # Ensure burnout is between 0 and 1

    def predict_optimal_conditions(self, weekly_hours: float, weekly_tasks: int):
        recent_data = np.array([[weekly_hours, weekly_tasks]])
        recent_data = self.scaler.transform(recent_data)
        optimal_hours = self.hours_model.predict(recent_data)[0]
        optimal_workload = self.workload_model.predict(recent_data)[0]
        return optimal_hours, optimal_workload

    def get_recommendations(self, weekly_hours: float, weekly_tasks: int):
        predicted_burnout = self.predict_burnout(weekly_hours, weekly_tasks)
        optimal_hours, optimal_workload = self.predict_optimal_conditions(weekly_hours, weekly_tasks)

        recommendations = []

        if weekly_hours > optimal_hours:
            recommendations.append(f"Reduce weekly working hours from {weekly_hours:.1f} to approximately {optimal_hours:.1f} hours")
        elif weekly_hours < optimal_hours:
            recommendations.append(f"You can potentially increase your weekly hours from {weekly_hours:.1f} to approximately {optimal_hours:.1f} hours without risking burnout")

        if weekly_tasks > optimal_workload:
            recommendations.append(f"Reduce your weekly tasks from {weekly_tasks} to approximately {optimal_workload:.0f} tasks")
        elif weekly_tasks < optimal_workload:
            recommendations.append(f"You can potentially handle more tasks. Consider increasing from {weekly_tasks} to approximately {optimal_workload:.0f} tasks per week")

        if predicted_burnout > 0.7:
            recommendations.append("Your predicted burnout rate is high. Consider taking some time off or reducing workload significantly")
        elif predicted_burnout > 0.5:
            recommendations.append("Your predicted burnout rate is moderate. Monitor closely and consider implementing stress-reduction techniques")

        return recommendations

def create_dashboard(bps: BurnoutPreventionSystem):
    st.title("Interactive Burnout Predictor")

    st.sidebar.header("Developer Information")
    name = st.sidebar.text_input("Name")
    role = st.sidebar.text_input("Role")

    # Train models and display MAE and MSE
    (burnout_mae, burnout_mse), (hours_mae, hours_mse), (workload_mae, workload_mse) = bps.train_models()
    st.write(f"Burnout Model - MAE: {burnout_mae:.4f}, MSE: {burnout_mse:.4f}")
    st.write(f"Hours Model - MAE: {hours_mae:.4f}, MSE: {hours_mse:.4f}")
    st.write(f"Workload Model - MAE: {workload_mae:.4f}, MSE: {workload_mse:.4f}")
    st.write("-------------------------------------------")
    st.write(f"Developer: {name}")
    st.write(f"Role: {role}")

    st.sidebar.header("Weekly Data")
    num_weeks = st.sidebar.number_input("Number of weeks to input", min_value=1, max_value=52, value=4)

    weeks_data = []
    for i in range(num_weeks):
        st.sidebar.subheader(f"Week {i+1}")
        week_start = st.sidebar.date_input(f"Week {i+1} Start Date", value=datetime.date.today() - datetime.timedelta(weeks=num_weeks-i-1))
        hours = st.sidebar.number_input(f"Hours Worked (Week {i+1})", min_value=0, max_value=168, value=40)
        tasks = st.sidebar.number_input(f"Number of Tasks (Week {i+1})", min_value=0, max_value=50, value=5)
        weeks_data.append({"week_start": week_start, "hours": hours, "tasks": tasks})

    if st.sidebar.button("Update Dashboard"):
        # Predict burnout rates
        dates = [week["week_start"] for week in weeks_data]
        hours = [week["hours"] for week in weeks_data]
        tasks = [week["tasks"] for week in weeks_data]
        burnout = [bps.predict_burnout(h, t) for h, t in zip(hours, tasks)]

        # Create time series plots
        fig_hours = go.Figure()
        fig_hours.add_trace(go.Scatter(x=dates, y=hours, mode='lines+markers', name='Hours Worked'))
        fig_hours.update_layout(title='Hours Worked Over Time', xaxis_title='Week Start Date', yaxis_title='Hours')
        st.plotly_chart(fig_hours)

        fig_burnout = go.Figure()
        fig_burnout.add_trace(go.Scatter(x=dates, y=burnout, mode='lines+markers', name='Predicted Burnout Rate'))
        fig_burnout.update_layout(title='Predicted Burnout Rate Over Time', xaxis_title='Week Start Date', yaxis_title='Burnout Rate')
        st.plotly_chart(fig_burnout)

        # Display task information
        st.subheader("Tasks Overview")
        task_df = pd.DataFrame({"Week Start": dates, "Number of Tasks": tasks})
        st.write(task_df)

        # Calculate weekly averages for recommendations
        avg_weekly_hours = np.mean(hours)
        avg_weekly_tasks = np.mean(tasks)

        # Display recommendations
        st.subheader("Recommendations")
        recommendations = bps.get_recommendations(avg_weekly_hours, avg_weekly_tasks)
        for rec in recommendations:
            st.write(f"- {rec}")

# Example usage
if __name__ == "__main__":
    bps = BurnoutPreventionSystem('developer_data.csv')
    create_dashboard(bps)