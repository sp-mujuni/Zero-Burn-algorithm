import datetime
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

class BurnoutPreventionSystem:
    def __init__(self):
        self.developers: List[Developer] = []
        self.hours_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.workload_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()

    def add_developer(self, developer: Developer):
        self.developers.append(developer)

    def log_work(self, developer: Developer, date: datetime.date, hours: float, burnout_rate: float):
        developer.hours_worked[date] = hours
        developer.burnout_rate[date] = burnout_rate

    def calculate_weekly_stats(self, developer: Developer, date: datetime.date):
        week_start = date - datetime.timedelta(days=date.weekday())
        week_end = week_start + datetime.timedelta(days=6)
        weekly_hours = sum(hours for day, hours in developer.hours_worked.items()
                           if week_start <= day <= week_end)
        weekly_burnout = np.mean([rate for day, rate in developer.burnout_rate.items()
                                  if week_start <= day <= week_end])
        weekly_tasks = len([task for task in developer.tasks
                            if week_start <= task['deadline'] <= week_end])
        return weekly_hours, weekly_burnout, weekly_tasks

    def prepare_data(self, developer: Developer):
        X, y_hours, y_workload = [], [], []
        dates = sorted(developer.hours_worked.keys())
        for i, date in enumerate(dates):
            if i >= 4:  # Use last 4 weeks of data to predict
                weekly_hours, weekly_burnout, weekly_tasks = self.calculate_weekly_stats(developer, date)
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

# Example usage
if __name__ == "__main__":
    bps = BurnoutPreventionSystem()

    # Add developers
    dev_name = input("Enter Developer Name: ")
    dev_role = input("Enter Developer Role: ")
    dev = Developer(dev_name, dev_role)
    bps.add_developer(dev)

    # Simulate data for a few months
    for week in range(20):
        for day in range(7):
            date = datetime.date.today() - datetime.timedelta(weeks=week, days=day)
            dev_hours = np.random.randint(6, 11)
            dev_burnout = min(1, max(0, np.random.normal(0.5, 0.2)))
            bps.log_work(dev, date, dev_hours, dev_burnout)

    dev_tasks = np.random.randint(5, 15)
    for _ in range(dev_tasks):
            dev.tasks.append({"name": "Task", "deadline": date + datetime.timedelta(days=7)})

    # Monitor burnout and provide recommendations
    bps.monitor_burnout()