import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, date, timedelta
import random
from typing import List, Dict
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class Developer:
    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role
        self.tasks: List[Dict] = []
        self.hours_worked: Dict[datetime.date, float] = {}
        self.burnout_rate: Dict[datetime.date, float] = {}  # 0-1 scale, 1 being complete burnout

    def log_work(self, date: datetime.date, hours: float, burnout_rate: float):
        self.hours_worked[date] = hours
        self.burnout_rate[date] = burnout_rate

class BurnoutPreventionSystem:
    def __init__(self):
        self.developers: List[Developer] = []
        self.hours_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.workload_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()

    def add_developer(self, developer: Developer):
        self.developers.append(developer)

    def log_work(self, developer: Developer, work_date: date, hours: float, burnout_rate: float):
        developer.log_work(work_date, hours, burnout_rate)

    def calculate_weekly_stats(self, developer: Developer, work_date: date):
        week_start = work_date - timedelta(days=work_date.weekday())
        week_end = week_start + timedelta(days=6)
        weekly_hours = sum(hours for day, hours in developer.hours_worked.items() if week_start <= day <= week_end)
        weekly_burnout = np.mean([rate for day, rate in developer.burnout_rate.items() if week_start <= day <= week_end])
        weekly_tasks = len([task for task in developer.tasks if week_start <= task['deadline'] <= week_end])
        return weekly_hours, weekly_burnout, weekly_tasks

    def prepare_data(self, developer: Developer):
        X, y_hours, y_workload = [], [], []
        dates = sorted(developer.hours_worked.keys())
        for i, work_date in enumerate(dates):
            if i >= 4:  # Use last 4 weeks of data to predict
                weekly_hours, weekly_burnout, weekly_tasks = self.calculate_weekly_stats(developer, work_date)
                features = [
                    np.mean([self.calculate_weekly_stats(developer, dates[j])[0] for j in range(i-4, i)]),
                    np.mean([self.calculate_weekly_stats(developer, dates[j])[1] for j in range(i-4, i)]),
                    np.mean([self.calculate_weekly_stats(developer, dates[j])[2] for j in range(i-4, i)])
                ]
                X.append(features)
                y_hours.append(weekly_hours)
                y_workload.append(weekly_tasks)
        return np.array(X), np.array(y_hours), np.array(y_workload)

    def train_models(self, developer: Developer):
        X, y_hours, y_workload = self.prepare_data(developer)
        if len(X) > 0:
            X = self.scaler.fit_transform(X)
            X_train, X_test, y_hours_train, y_hours_test = train_test_split(X, y_hours, test_size=0.2, random_state=42)
            self.hours_model.fit(X_train, y_hours_train)
            print(f"Hours model R2 score for {developer.name}: {self.hours_model.score(X_test, y_hours_test):.2f}")

            X_train, X_test, y_workload_train, y_workload_test = train_test_split(X, y_workload, test_size=0.2, random_state=42)
            self.workload_model.fit(X_train, y_workload_train)
            print(f"Workload model R2 score for {developer.name}: {self.workload_model.score(X_test, y_workload_test):.2f}")

    def predict_optimal_conditions(self, developer: Developer):
        recent_data = self.prepare_data(developer)[0][-1].reshape(1, -1)
        recent_data = self.scaler.transform(recent_data)
        optimal_hours = self.hours_model.predict(recent_data)[0]
        optimal_workload = self.workload_model.predict(recent_data)[0]
        return optimal_hours, optimal_workload

    def get_recommendations(self, developer: Developer):
        self.train_models(developer)
        optimal_hours, optimal_workload = self.predict_optimal_conditions(developer)

        recent_weekly_hours, recent_burnout, recent_tasks = self.calculate_weekly_stats(developer, max(developer.hours_worked.keys()))

        recommendations = []

        if recent_weekly_hours > optimal_hours:
            recommendations.append(f"Reduce weekly working hours from {recent_weekly_hours:.1f} to approximately {optimal_hours:.1f} hours")
        elif recent_weekly_hours < optimal_hours:
            recommendations.append(f"You can potentially increase your weekly hours from {recent_weekly_hours:.1f} to approximately {optimal_hours:.1f} hours without risking burnout")

        if recent_tasks > optimal_workload:
            recommendations.append(f"Reduce your weekly tasks from {recent_tasks} to approximately {optimal_workload:.0f} tasks")
        elif recent_tasks < optimal_workload:
            recommendations.append(f"You can potentially handle more tasks. Consider increasing from {recent_tasks} to approximately {optimal_workload:.0f} tasks per week")

        if recent_burnout > 0.7:
            recommendations.append("Your recent burnout rate is high. Consider taking some time off or reducing workload significantly")
        elif recent_burnout > 0.5:
            recommendations.append("Your burnout rate is moderate. Monitor closely and consider implementing stress-reduction techniques")

        return recommendations

    def monitor_burnout(self):
        for developer in self.developers:
            recommendations = self.get_recommendations(developer)
            print(f"\nRecommendations for {developer.name}:")
            for recommendation in recommendations:
                print(f"- {recommendation}")

# Helper function to generate sample data
def generate_sample_data(developer: Developer, weeks: int):
    end_date = date.today()
    start_date = end_date - timedelta(weeks=weeks)
    current_date = start_date
    while current_date <= end_date:
        hours = random.uniform(6, 11)
        burnout_rate = min(1, max(0, random.normalvariate(0.5, 0.2)))
        developer.log_work(current_date, hours, burnout_rate)
        if current_date.weekday() == 6:  # Sunday
            tasks = random.randint(5, 15)
            for _ in range(tasks):
                developer.tasks.append({"name": "Task", "deadline": current_date + timedelta(days=7)})
        current_date += timedelta(days=1)

# Dashboard
def main():
    st.set_page_config(page_title="Developer Burnout Prevention Dashboard", layout="wide")
    st.title("Developer Burnout Prevention Dashboard")

    # Initialize session state
    if 'bps' not in st.session_state:
        st.session_state.bps = BurnoutPreventionSystem()
        alice = Developer("Alice", "Frontend Developer")
        bob = Developer("Bob", "Backend Developer")
        st.session_state.bps.add_developer(alice)
        st.session_state.bps.add_developer(bob)
        generate_sample_data(alice, 12)
        generate_sample_data(bob, 12)

    # Sidebar
    st.sidebar.title("Settings")
    selected_developer = st.sidebar.selectbox("Select Developer", [dev.name for dev in st.session_state.bps.developers])
    date_range = st.sidebar.slider("Date Range (weeks)", 1, 12, 4)

    # Main content
    developer = next(dev for dev in st.session_state.bps.developers if dev.name == selected_developer)
    
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Burnout Rate Over Time")
        df_burnout = pd.DataFrame(developer.burnout_rate.items(), columns=['Date', 'Burnout Rate'])
        df_burnout = df_burnout.sort_values('Date').reset_index(drop=True)
        df_burnout = df_burnout.tail(date_range * 7)
        fig_burnout = px.line(df_burnout, x='Date', y='Burnout Rate', title=f"{developer.name}'s Burnout Rate")
        st.plotly_chart(fig_burnout)

    with col2:
        st.subheader("Weekly Hours Worked")
        weekly_hours = {}
        for date, hours in developer.hours_worked.items():
            week_start = date - timedelta(days=date.weekday())
            if week_start not in weekly_hours:
                weekly_hours[week_start] = 0
            weekly_hours[week_start] += hours
        df_hours = pd.DataFrame(weekly_hours.items(), columns=['Week', 'Hours'])
        df_hours = df_hours.sort_values('Week').reset_index(drop=True)
        df_hours = df_hours.tail(date_range)
        fig_hours = px.bar(df_hours, x='Week', y='Hours', title=f"{developer.name}'s Weekly Hours")
        st.plotly_chart(fig_hours)

    st.subheader("Recommendations")
    recommendations = st.session_state.bps.get_recommendations(developer)
    for rec in recommendations:
        st.write(f"- {rec}")

    st.subheader("Current Week Stats")
    current_week_start = date.today() - timedelta(days=date.today().weekday())
    current_week_hours = sum(hours for date, hours in developer.hours_worked.items() if date >= current_week_start)
    current_week_tasks = len([task for task in developer.tasks if task['deadline'] >= current_week_start])
    current_week_burnout = np.mean([rate for date, rate in developer.burnout_rate.items() if date >= current_week_start])

    col1, col2, col3 = st.columns(3)
    col1.metric("Current Week Hours", f"{current_week_hours:.1f}")
    col2.metric("Current Week Tasks", current_week_tasks)
    col3.metric("Current Week Avg Burnout", f"{current_week_burnout:.2f}")

    st.subheader("Task List")
    task_df = pd.DataFrame(developer.tasks)
    if not task_df.empty:
        task_df['deadline'] = pd.to_datetime(task_df['deadline'])
        task_df = task_df.sort_values('deadline')
        st.dataframe(task_df)
    else:
        st.write("No tasks currently assigned.")

if __name__ == "__main__":
    main()