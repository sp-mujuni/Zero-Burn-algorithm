import datetime
from typing import List, Dict
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class Developer:
    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role
        self.tasks: List[Dict] = []
        self.hours_worked: Dict[datetime.date, float] = {}
        self.stress_level: int = 0
        self.burnout_history: List[int] = []  # 0 for no burnout, 1 for burnout

class BurnoutPreventionSystem:
    def __init__(self):
        self.developers: List[Developer] = []
        self.max_weekly_hours: float = 40
        self.max_daily_hours: float = 8
        self.max_stress_level: int = 7
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()

    def add_developer(self, developer: Developer):
        self.developers.append(developer)

    def log_work(self, developer: Developer, date: datetime.date, hours: float):
        if date not in developer.hours_worked:
            developer.hours_worked[date] = 0
        developer.hours_worked[date] += hours

    def add_task(self, developer: Developer, task: Dict):
        developer.tasks.append(task)

    def calculate_weekly_hours(self, developer: Developer, date: datetime.date) -> float:
        week_start = date - datetime.timedelta(days=date.weekday())
        week_end = week_start + datetime.timedelta(days=6)
        return sum(hours for day, hours in developer.hours_worked.items() 
                   if week_start <= day <= week_end)

    def update_stress_level(self, developer: Developer):
        weekly_hours = self.calculate_weekly_hours(developer, datetime.date.today())
        task_count = len(developer.tasks)
        
        if weekly_hours > self.max_weekly_hours or task_count > 10:
            developer.stress_level += 1
        elif weekly_hours < 30 and task_count < 5:
            developer.stress_level = max(0, developer.stress_level - 1)

    def prepare_data(self):
        X = []
        y = []
        for dev in self.developers:
            for i in range(len(dev.burnout_history)):
                if i >= 4:  # Use last 4 weeks of data to predict burnout
                    features = [
                        np.mean([self.calculate_weekly_hours(dev, datetime.date.today() - datetime.timedelta(weeks=w)) for w in range(1, 5)]),
                        np.mean([len(dev.tasks) for _ in range(4)]),
                        np.mean(dev.burnout_history[i-4:i]),
                        dev.stress_level
                    ]
                    X.append(features)
                    y.append(dev.burnout_history[i])
        return np.array(X), np.array(y)

    def train_model(self):
        X, y = self.prepare_data()
        if len(X) > 0:
            X = self.scaler.fit_transform(X)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            self.model.fit(X_train, y_train)
            print(f"Model accuracy: {self.model.score(X_test, y_test):.2f}")

    def predict_burnout_risk(self, developer: Developer) -> float:
        features = [
            np.mean([self.calculate_weekly_hours(developer, datetime.date.today() - datetime.timedelta(weeks=w)) for w in range(1, 5)]),
            len(developer.tasks),
            np.mean(developer.burnout_history[-4:]) if len(developer.burnout_history) >= 4 else 0,
            developer.stress_level
        ]
        features = self.scaler.transform([features])
        return self.model.predict_proba(features)[0][1]  # Probability of burnout

    def get_recommendations(self, developer: Developer) -> List[str]:
        recommendations = []
        weekly_hours = self.calculate_weekly_hours(developer, datetime.date.today())
        burnout_risk = self.predict_burnout_risk(developer)
        
        if weekly_hours > self.max_weekly_hours:
            recommendations.append("Reduce working hours to prevent overwork")
        
        if developer.stress_level > self.max_stress_level:
            recommendations.append("Take a day off to recover from high stress")
        
        if len(developer.tasks) > 10:
            recommendations.append("Delegate or postpone some tasks to reduce workload")
        
        if burnout_risk > 0.7:
            recommendations.append("High risk of burnout detected. Consider taking a longer break or vacation")
        elif burnout_risk > 0.5:
            recommendations.append("Moderate risk of burnout. Implement stress-reduction techniques and reassess workload")

        return recommendations

    def monitor_burnout(self):
        self.train_model()  # Train the model before making predictions
        for developer in self.developers:
            self.update_stress_level(developer)
            recommendations = self.get_recommendations(developer)
            burnout_risk = self.predict_burnout_risk(developer)
            
            print(f"Burnout risk for {developer.name}: {burnout_risk:.2f}")
            if recommendations:
                print(f"Recommendations for {developer.name}:")
                for recommendation in recommendations:
                    print(f"- {recommendation}")
            else:
                print(f"{developer.name} is doing well. No recommendations at this time.")

# Example usage
if __name__ == "__main__":
    bps = BurnoutPreventionSystem()

    # Add developers
    alice = Developer("Alice", "Frontend Developer")
    bob = Developer("Bob", "Backend Developer")
    bps.add_developer(alice)
    bps.add_developer(bob)

    # Simulate data for a few weeks
    for week in range(10):
        for day in range(7):
            date = datetime.date.today() - datetime.timedelta(weeks=week, days=day)
            bps.log_work(alice, date, np.random.randint(6, 11))
            bps.log_work(bob, date, np.random.randint(6, 11))
        
        alice.burnout_history.append(np.random.choice([0, 1], p=[0.8, 0.2]))
        bob.burnout_history.append(np.random.choice([0, 1], p=[0.9, 0.1]))

        for _ in range(np.random.randint(5, 15)):
            bps.add_task(alice, {"name": "Task", "deadline": date + datetime.timedelta(days=7)})
        
        for _ in range(np.random.randint(3, 10)):
            bps.add_task(bob, {"name": "Task", "deadline": date + datetime.timedelta(days=7)})

    # Monitor burnout
    bps.monitor_burnout()