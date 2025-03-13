import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('TkAgg')

class EmotionAnalysisTime:
    def __init__(self, combined_df):
        """
        Initialize the class with the provided data.
        """
        self.combined_df = combined_df

        # Convert 'created_utc' column to datetime format (if not already in datetime format)
        self.combined_df['created_utc'] = pd.to_datetime(self.combined_df['created_utc'], errors='coerce')

        # Extract day of the week (0 = Sunday, 6 = Saturday)
        self.combined_df.loc[:, 'day_of_week'] = self.combined_df['created_utc'].dt.dayofweek

        # Extract hour of the day (0-23)
        self.combined_df.loc[:, 'hour_of_day'] = self.combined_df['created_utc'].dt.hour

        # Extract day of the month
        self.combined_df.loc[:, 'day_of_month'] = self.combined_df['created_utc'].dt.day

    def add_time_of_day_column(self):
        """
        Add a 'time_of_day' column to the DataFrame based on the 'hour_of_day' column.
        """
        self.combined_df.loc[:, 'time_of_day'] = self.combined_df['hour_of_day'].apply(self.categorize_time_of_day)

    def categorize_time_of_day(self, hour):
        """
        Categorize the time of day based on the hour of the day.
        """
        if 6 <= hour < 12:
            return 'Morning'
        elif 12 <= hour < 18:
            return 'Afternoon'
        elif 18 <= hour < 24:
            return 'Evening'
        else:
            return 'Night'

    def analyze_specific_emotions(self, emotion_columns):
        """
        Analyze specific emotions provided in a list.
        """
        for emotion in emotion_columns:
            if emotion in self.combined_df.columns:
                print(f"Analyzing emotion: {emotion}")
                self.analyze_emotion_over_time(emotion)
            else:
                print(f"Emotion '{emotion}' not found in the dataset.")

    def analyze_all_emotions(self):
        """
        Analyze all emotions collectively in the DataFrame.
        """
        # Identify emotion columns that are numeric only
        emotion_columns = [
            col for col in self.combined_df.columns
            if col not in ['created_utc', 'author', 'day_of_week', 'hour_of_day', 'day_of_month', 'time_of_day']
               and pd.api.types.is_numeric_dtype(self.combined_df[col])
        ]

        # Validate that emotion_columns are properly filtered
        if not emotion_columns:
            raise ValueError("No numeric emotion columns found in the dataset.")

        # Create a new column for the sum of all emotions in each row
        self.combined_df['total_emotions'] = self.combined_df[emotion_columns].sum(axis=1)

        # Analyze emotions by time of day
        self.combined_df['time_of_day'] = self.combined_df['created_utc'].dt.hour.apply(self.categorize_time_of_day)
        time_of_day_summary = self.combined_df.groupby('time_of_day')['total_emotions'].sum()

        plt.figure(figsize=(10, 6))
        time_of_day_summary.plot(kind='bar', color='skyblue')
        plt.title("Total Emotions by Time of Day")
        plt.xlabel("Time of Day (Morning, Noon, Evening, Night)")
        plt.ylabel("Total Count of Emotions")
        plt.xticks(rotation=0)
        plt.show()

        # Analyze emotions by day of the week
        self.combined_df['day_of_week'] = self.combined_df['created_utc'].dt.dayofweek
        day_of_week_summary = self.combined_df.groupby('day_of_week')['total_emotions'].sum()

        plt.figure(figsize=(10, 6))
        day_of_week_summary.plot(kind='bar', color='lightgreen')
        plt.title("Total Emotions by Day of the Week")
        plt.xlabel("Day of the Week")
        plt.ylabel("Total Count of Emotions")
        plt.show()

        # Analyze emotions by day of the month
        self.combined_df['day_of_month'] = self.combined_df['created_utc'].dt.day
        day_of_month_summary = self.combined_df.groupby('day_of_month')['total_emotions'].sum()

        plt.figure(figsize=(10, 6))
        day_of_month_summary.plot(kind='bar', color='lightcoral')
        plt.title("Total Emotions by Day of the Month")
        plt.xlabel("Day of the Month")
        plt.ylabel("Total Count of Emotions")
        plt.xticks(rotation=0)
        plt.show()

        print("Analysis of all emotions collectively has been completed.")

    def analyze_emotion_over_time(self, emotion_column):
        """
        Analyze the given emotion over time and plot the frequency distribution.
        """
        # Filter the DataFrame based on the selected emotion
        emotion_data = self.combined_df[self.combined_df[emotion_column] == 1].copy()

        # Add time of day column to the DataFrame
        emotion_data.loc[:, 'time_of_day'] = emotion_data['created_utc'].dt.hour.apply(self.categorize_time_of_day)
        time_of_day_frequency = emotion_data.groupby('time_of_day')['author'].count()

        # Plot frequency by time of day
        plt.figure(figsize=(10, 6))
        time_of_day_frequency.plot(kind='bar', color='skyblue')
        plt.title(f"Distribution of {emotion_column} by Time of Day")
        plt.xlabel("Time of Day (Morning, Noon, Evening, Night)")
        plt.ylabel("Number of Comments")
        plt.xticks(rotation=0)
        plt.show()

        # Calculate and plot frequency by day of week
        emotion_data.loc[:, 'day_of_week'] = emotion_data['created_utc'].dt.dayofweek
        day_of_week_frequency = emotion_data.groupby('day_of_week')['author'].count()
        plt.figure(figsize=(10, 6))
        day_of_week_frequency.plot(kind='bar', color='lightgreen')
        plt.title(f"Distribution of {emotion_column} by Day of Week")
        plt.xlabel("Day of the Week")
        plt.ylabel("Number of Comments")
        plt.show()

        # Calculate and plot frequency by day of month
        emotion_data.loc[:, 'day_of_month'] = emotion_data['created_utc'].dt.day
        day_of_month_frequency = emotion_data.groupby('day_of_month')['author'].count()
        plt.figure(figsize=(10, 6))
        day_of_month_frequency.plot(kind='bar', color='lightcoral')
        plt.title(f"Distribution of {emotion_column} by Day of Month")
        plt.xlabel("Day of the Month")
        plt.ylabel("Number of Comments")
        plt.xticks(rotation=0)
        plt.show()


def main():
    # Load your combined dataframe (replace with your actual dataframe)
    combined_df = pd.read_csv('data/full_dataset/goemotions_combined.csv')

    # Create an instance of the EmotionAnalysisTime class
    emotion_analysis = EmotionAnalysisTime(combined_df)

    # Analyze specific emotions (e.g., love and joy)


    # Analyze all emotions
    emotion_analysis.analyze_all_emotions()
    emotion_analysis.analyze_specific_emotions(['love', 'joy'])
if __name__ == "__main__":
    main()
