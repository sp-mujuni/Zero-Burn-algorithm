import datetime
from typing import List, Dict

class Developer:
    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role
        self.tasks: List[Dict] = []
        self.hours_worked: Dict[datetime.date, float] = {}
        self.stress_level: int = 0

class BurnoutPreventionSystem:
    def __init__(self):
        self.developers: List[Developer] = []
        self.max_weekly_hours: float = 40
        self.max_daily_hours: float = 8
        self.max_stress_level: int = 7

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

    def get_recommendations(self, developer: Developer) -> List[str]:
        recommendations = []
        weekly_hours = self.calculate_weekly_hours(developer, datetime.date.today())
        
        if weekly_hours > self.max_weekly_hours:
            recommendations.append("Reduce working hours to prevent overwork")
        
        if developer.stress_level > self.max_stress_level:
            recommendations.append("Take a day off to recover from high stress")
        
        if len(developer.tasks) > 10:
            recommendations.append("Delegate or postpone some tasks to reduce workload")
        
        return recommendations

    def monitor_burnout(self):
        for developer in self.developers:
            self.update_stress_level(developer)
            recommendations = self.get_recommendations(developer)
            
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

    # Log work and add tasks
    today = datetime.date.today()
    bps.log_work(alice, today, 10)
    bps.log_work(bob, today, 8)
    
    for _ in range(12):
        bps.add_task(alice, {"name": "Task", "deadline": today + datetime.timedelta(days=7)})
    
    for _ in range(5):
        bps.add_task(bob, {"name": "Task", "deadline": today + datetime.timedelta(days=7)})

    # Monitor burnout
    bps.monitor_burnout()