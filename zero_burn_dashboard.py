import datetime
from typing import List, Dict
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

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
        self.hours_model = LinearRegression()
        self.workload_model = LinearRegression()
        self.scaler = StandardScaler()
        self.data = pd.read_csv('developer_data.csv')

    def prepare_data(self):
        X = self.data[['weekly_hours', 'weekly_burnout', 'weekly_tasks']]
        y_hours = self.data['next_week_hours']
        y_workload = self.data['next_week_tasks']
        return X, y_hours, y_workload

    def train_models(self):
        X, y_hours, y_workload = self.prepare_data()
        X = self.scaler.fit_transform(X)

        X_train, X_test, y_hours_train, y_hours_test = train_test_split(X, y_hours, test_size=0.2, random_state=42)
        self.hours_model.fit(X_train, y_hours_train)
        hours_pred = self.hours_model.predict(X_test)
        hours_mse = mean_squared_error(y_hours_test, hours_pred)

        X_train, X_test, y_workload_train, y_workload_test = train_test_split(X, y_workload, test_size=0.2, random_state=42)
        self.workload_model.fit(X_train, y_workload_train)
        workload_pred = self.workload_model.predict(X_test)
        workload_mse = mean_squared_error(y_workload_test, workload_pred)

        return hours_mse, workload_mse

    def predict_optimal_conditions(self, weekly_hours: float, weekly_burnout: float, weekly_tasks: int):
        recent_data = np.array([[weekly_hours, weekly_burnout, weekly_tasks]])
        recent_data = self.scaler.transform(recent_data)
        optimal_hours = self.hours_model.predict(recent_data)[0]
        optimal_workload = self.workload_model.predict(recent_data)[0]
        return optimal_hours, optimal_workload

    def get_recommendations(self, weekly_hours: float, weekly_burnout: float, weekly_tasks: int):
        optimal_hours, optimal_workload = self.predict_optimal_conditions(weekly_hours, weekly_burnout, weekly_tasks)

        recommendations = []

        if weekly_hours > optimal_hours:
            recommendations.append(f"Reduce weekly working hours from {weekly_hours:.1f} to approximately {optimal_hours:.1f} hours")
        elif weekly_hours < optimal_hours:
            recommendations.append(f"You can potentially increase your weekly hours from {weekly_hours:.1f} to approximately {optimal_hours:.1f} hours without risking burnout")

        if weekly_tasks > optimal_workload:
            recommendations.append(f"Reduce your weekly tasks from {weekly_tasks} to approximately {optimal_workload:.0f} tasks")
        elif weekly_tasks < optimal_workload:
            recommendations.append(f"You can potentially handle more tasks. Consider increasing from {weekly_tasks} to approximately {optimal_workload:.0f} tasks per week")

        if weekly_burnout > 0.7:
            recommendations.append("Your recent burnout rate is high. Consider taking some time off or reducing workload significantly")
        elif weekly_burnout > 0.5:
            recommendations.append("Your burnout rate is moderate. Monitor closely and consider implementing stress-reduction techniques")

        return recommendations

def create_dashboard(bps: BurnoutPreventionSystem):
    st.title("Interactive Burnout Tracker")

    global name, role    
    st.sidebar.header("Developer Information")
    name = st.sidebar.text_input("Name")
    role = st.sidebar.text_input("Role")

    # Train models and display MSE
    hours_mse, workload_mse = bps.train_models()
    st.write(f"Hours Model MSE: {hours_mse:.2f}")
    st.write(f"Workload Model MSE: {workload_mse:.2f}")
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
        burnout = st.sidebar.slider(f"Burnout Rate (Week {i+1})", 0.0, 1.0, 0.5)
        weeks_data.append({"week_start": week_start, "hours": hours, "burnout": burnout})

    st.sidebar.header("Tasks")
    num_tasks = st.sidebar.number_input("Number of tasks", min_value=0, max_value=20, value=5)

    tasks = []
    for i in range(num_tasks):
        st.sidebar.subheader(f"Task {i+1}")
        task_name = st.sidebar.text_input(f"Task {i+1} Name", value=f"Task {i+1}")
        deadline = st.sidebar.date_input(f"Task {i+1} Deadline", value=datetime.date.today() + datetime.timedelta(days=7))
        estimated_hours = st.sidebar.number_input(f"Task {i+1} Estimated Hours", min_value=1, max_value=168, value=8)
        tasks.append({"name": task_name, "deadline": deadline, "estimated_hours": estimated_hours})

    if st.sidebar.button("Update Dashboard"):
        # Create time series plots
        dates = [week["week_start"] for week in weeks_data]
        hours = [week["hours"] for week in weeks_data]
        burnout = [week["burnout"] for week in weeks_data]

        fig_hours = go.Figure()
        fig_hours.add_trace(go.Scatter(x=dates, y=hours, mode='lines+markers', name='Hours Worked'))
        fig_hours.update_layout(title='Hours Worked Over Time', xaxis_title='Week Start Date', yaxis_title='Hours')
        st.plotly_chart(fig_hours)

        fig_burnout = go.Figure()
        fig_burnout.add_trace(go.Scatter(x=dates, y=burnout, mode='lines+markers', name='Burnout Rate'))
        fig_burnout.update_layout(title='Burnout Rate Over Time', xaxis_title='Week Start Date', yaxis_title='Burnout Rate')
        st.plotly_chart(fig_burnout)

        # Display task information
        st.subheader("Current Tasks")
        task_df = pd.DataFrame(tasks)
        st.write(task_df)

        # Calculate weekly averages for recommendations
        avg_weekly_hours = np.mean(hours)
        avg_weekly_burnout = np.mean(burnout)
        avg_weekly_tasks = len([task for task in tasks if task['deadline'] <= max(dates) + datetime.timedelta(days=7)])

        # Display recommendations
        st.subheader("Recommendations")
        recommendations = bps.get_recommendations(avg_weekly_hours, avg_weekly_burnout, avg_weekly_tasks)
        for rec in recommendations:
            st.write(f"- {rec}")

        # # Interactive prediction
        # st.subheader("Predict Optimal Conditions")
        # predict_weekly_hours = st.slider("Weekly Hours", 0, 80, int(avg_weekly_hours))
        # predict_weekly_burnout = st.slider("Weekly Burnout Rate", 0.0, 1.0, float(avg_weekly_burnout))
        # predict_weekly_tasks = st.slider("Weekly Tasks", 0, 20, int(avg_weekly_tasks))

        # if st.button("Predict"):
        #     optimal_hours, optimal_workload = bps.predict_optimal_conditions(predict_weekly_hours, predict_weekly_burnout, predict_weekly_tasks)
        #     st.write(f"Predicted Optimal Hours: {optimal_hours:.2f}")
        #     st.write(f"Predicted Optimal Workload: {optimal_workload:.2f}")

# Example usage
if __name__ == "__main__":
    bps = BurnoutPreventionSystem('developer_data.csv')
    create_dashboard(bps)