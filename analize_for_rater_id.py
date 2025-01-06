import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import matplotlib
import re
from collections import Counter
from collections import defaultdict

# matplotlib.use('TkAgg')





class rater_id_EmotionAnalysis:
    def __init__(self, combined_df):
        """
        Initialize the analysis class for rater_id-related statistics.

        Args:
            combined_df (pd.DataFrame): The dataset to analyze.
        """
        self.combined_df = combined_df
        self.combined_df['created_utc'] = pd.to_datetime(self.combined_df['created_utc'], errors='coerce')
        self.emotion_columns = combined_df.columns[9:]
        self.rater_summary = self._calculate_rater_summary()

    def _calculate_rater_summary(self):
        """
        Summarize the emotions annotated by each rater.

        Returns:
            pd.DataFrame: Summary of total emotions annotated by each rater.
        """
        rater_summary = self.combined_df.groupby('rater_id')[self.emotion_columns].sum()
        rater_summary['total_annotations'] = rater_summary.sum(axis=1)
        rater_summary['unique_emotions'] = (rater_summary[self.emotion_columns] > 0).sum(axis=1)
        return rater_summary

    def summarize_rater_contributions(self):
        """
        Display and return the number of annotations per rater.
        """
        rater_counts = self.combined_df['rater_id'].value_counts()
        print("Number of annotations per rater:")
        print(rater_counts)
        return rater_counts


    def _analyze_annotations_over_time(self, rater_data, rater_id):
        """
        Analyze and plot the rater's annotations over time.

        Args:
            rater_data (pd.DataFrame): Filtered data for the specific rater.
            rater_id (str): The rater's ID.
        """
        rater_data['created_date'] = rater_data['created_utc'].dt.date
        comments_by_date = rater_data['created_date'].value_counts().sort_index()

        # Plot annotations over time
        plt.figure(figsize=(12, 6))
        comments_by_date.plot(kind='bar', color='orange', edgecolor='black')
        plt.title(f"Annotations Over Time for Rater ID: {rater_id}")
        plt.xlabel("Date")
        plt.ylabel("Number of Annotations")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def visualize_rater_diversity(self):
        """
        Visualize the diversity of emotions annotated by raters.
        """
        plt.figure(figsize=(12, 6))
        self.rater_summary['unique_emotions'].plot(kind='bar', color='green', edgecolor='black')
        plt.title("Diversity of Emotions Annotated by Raters")
        plt.xlabel("Rater ID")
        plt.ylabel("Number of Unique Emotions")
        plt.tight_layout()
        plt.show()

    def visualize_rater_totals(self):
        """
        Visualize the total number of annotations by each rater.
        """
        plt.figure(figsize=(12, 6))
        self.rater_summary['total_annotations'].plot(kind='bar', color='blue', edgecolor='black')
        plt.title("Total Annotations by Raters")
        plt.xlabel("Rater ID")
        plt.ylabel("Number of Annotations")
        plt.tight_layout()
        plt.show()


    def analyze_specific_rater(self, rater_id):
        """
        Perform a detailed analysis for a specific rater.

        Args:
            rater_id (str): The ID of the rater to analyze.

        Returns:
            None. Displays detailed analysis results and plots.
        """
        # Filter data for the specified rater
        rater_data = self.combined_df[self.combined_df['rater_id'] == rater_id]
        if rater_data.empty:
            print(f"No data found for rater ID: {rater_id}")
            return

        # Calculate the total emotion counts rated by the rater
        total_emotions = rater_data[self.emotion_columns].sum()
        print(f"\nTotal emotion counts for rater ID '{rater_id}':")
        print(total_emotions)

        # Identify the top 5 emotions rated by the rater
        top_emotions = total_emotions.sort_values(ascending=False).head(5)
        print(f"\nTop 5 emotions for rater ID '{rater_id}':")
        print(top_emotions)

        # Count the number of unique emotions rated at least once
        unique_emotions_count = (total_emotions > 0).sum()
        print(f"\nRater ID '{rater_id}' has rated {unique_emotions_count} unique emotions.")

        # Analyze the rater's activity over time
        self._analyze_annotations_over_time(rater_data, rater_id)

        # Visualize the total emotion counts as a bar chart
        plt.figure(figsize=(12, 6))
        total_emotions.plot(kind='bar', color='purple', edgecolor='black')
        plt.title(f"Emotion Counts for Rater ID: {rater_id}")
        plt.xlabel("Emotion")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.show()

        # Visualize the top 5 emotions as a bar chart
        plt.figure(figsize=(8, 5))
        top_emotions.plot(kind='bar', color='orange', edgecolor='black')
        plt.title(f"Top 5 Emotions for Rater ID: {rater_id}")
        plt.xlabel("Emotion")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.show()


# Main Function
def main():
    # Load the dataset (adjust file path)
    file_path = 'data/full_dataset/goemotions_combined.csv'
    combined_df = pd.read_csv(file_path)

    # Initialize class
    analysis = rater_id_EmotionAnalysis(combined_df)

    # Summarize contributions
    analysis.summarize_rater_contributions()

    # Analyze specific rater

    # # Visualize diversity and totals
    analysis.visualize_rater_diversity()
    analysis.visualize_rater_totals()
    # analysis.analyze_rater(rater_id=7)
    analysis.analyze_specific_rater(rater_id=61)


if __name__ == "__main__":
    main()