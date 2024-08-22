import datetime
from typing import List, Dict
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# import openai  # For GPT-4 integration

class Developer:
    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role
        self.tasks: List[Dict] = []
        self.hours_worked: Dict[datetime.date, float] = {}
        self.burnout_rate: Dict[datetime.date, float] = {}  # 0-1 scale, 1 being complete burnout
        self.ai_assistance_enabled = False  # Attribute for AI assistance

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


    def suggest_tasks(self, developer: Developer, date: datetime.date):
        week_start = date - datetime.timedelta(days=date.weekday())
        week_end = week_start + datetime.timedelta(days=6)
        
        # Filter tasks for the current week
        weekly_tasks = [task for task in developer.tasks if week_start <= task['deadline'] <= week_end]
        
        # Sort tasks by deadline and estimated hours
        sorted_tasks = sorted(weekly_tasks, key=lambda x: (x['deadline'], x['estimated_hours']))
        
        # Calculate available hours based on optimal conditions
        optimal_hours, _ = self.predict_optimal_conditions(developer)
        available_hours = optimal_hours - sum(developer.hours_worked.get(d, 0) for d in range(week_start, week_end + datetime.timedelta(days=1)))
        
        suggested_tasks = []
        for task in sorted_tasks:
            if available_hours >= task['estimated_hours']:
                suggested_tasks.append(task)
                available_hours -= task['estimated_hours']
            else:
                break
        
        return suggested_tasks

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

        # Add task suggestions
        today = datetime.date.today()
        suggested_tasks = self.suggest_tasks(developer, today)
        if suggested_tasks:
            recommendations.append("Suggested tasks for this week:")
            for task in suggested_tasks:
                recommendations.append(f"  - {task['name']} (Due: {task['deadline']}, Estimated hours: {task['estimated_hours']})")
        
        # Suggest AI assistance if burnout is high
        recent_burnout = self.calculate_weekly_stats(developer, max(developer.hours_worked.keys()))[1]
        if recent_burnout > 0.6 and not developer.ai_assistance_enabled:
            recommendations.append("Consider enabling AI assistance to help with coding tasks and reduce workload")
            # Toggle AI assistance
            bps.toggle_ai_assistance(dev)
            # Use AI assistance for a task
            activate_ai = input("Do you want to activate AI assistance? (y/n): ")
            if activate_ai.lower() == "y":
                while True:
                    task_index = int(input("Enter the task index to generate code for (0-9), type 10 to close the session: "))
                    if task_index == 10:
                        break
                    if 0 <= task_index < 10:
                        task = dev.tasks[task_index]
                        bps.use_ai_assistance(dev, task)
                    else:
                        print("Invalid task index. Please enter a number between 0 and 9.")
                            
            else:
                print("AI assistance not activated.")

        return recommendations
        
    
    def monitor_burnout(self):
        for developer in self.developers:
            recommendations = self.get_recommendations(developer)
            print(f"\nRecommendations for {developer.name}:")
            for recommendation in recommendations:
                print(f"- {recommendation}")

    def toggle_ai_assistance(self, developer: Developer):
        developer.ai_assistance_enabled = not developer.ai_assistance_enabled
        status = "disabled" if developer.ai_assistance_enabled else "enabled"
        print(f"AI assistance for {developer.name} is {status}")

    def use_ai_assistance(self, developer: Developer, task: Dict):
        if not developer.ai_assistance_enabled:
            print("AI assistance is not enabled for this developer")
            return

        # Placeholder for actual GPT-4 or Gemini integration
        # Set up the appropriate API keys and client
        prompt = f"Generate code for the following task: {task['name']}\nDetails: {task.get('description', 'No description provided')}"
        
        try:
            # Placeholder for API call
            # response = openai.Completion.create(engine="text-davinci-002", prompt=prompt, max_tokens=500)
            # generated_code = response.choices[0].text.strip()
            generated_code = "# Placeholder for generated code\n# Implement task logic here"
            
            print(f"AI-generated code for task '{task['name']}':")
            print(generated_code)
            print("\nDeveloper should review and refine the generated code.")
        except Exception as e:
            print(f"Error using AI assistance: {str(e)}")

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

    # Add some tasks
    for _ in range(10):
        deadline = datetime.date.today() + datetime.timedelta(days=np.random.randint(1, 14))
        dev.tasks.append({
            "name": f"Task {_ + 1}",
            "deadline": deadline,
            "estimated_hours": np.random.randint(2, 8),
            "description": f"Description for Task {_ + 1}"
        })

    # Monitor burnout and provide recommendations
    bps.monitor_burnout()