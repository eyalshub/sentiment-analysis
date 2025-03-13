import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import pi
import matplotlib
import seaborn as sns

# matplotlib.use('TkAgg')


class AuthorEmotionAnalysis:
    def __init__(self, dataframe):
        """
        Initializes the AuthorEmotionAnalysis class.

        Parameters:
        dataframe (pd.DataFrame): The dataframe containing the dataset.
        """
        self.df = dataframe.copy()
        self.df['created_utc'] = pd.to_datetime(self.df['created_utc'], errors='coerce')

        # Extracting emotion columns (assuming they start from column index 9)
        self.emotion_columns = self.df.columns[9:]
        self.author_emotions = self.df.groupby('author', as_index=True)[self.emotion_columns].sum()

        self.overall_emotions_avg = self.author_emotions[self.emotion_columns].mean()

    def get_authors(self):
        author_counts = self.df['author'].value_counts()
        return author_counts




    def get_emotion_distribution_for_author(self, author_id):
        """
        Returns the distribution of emotions for a specific author.

        Parameters:
        author_id (str): The ID of the author (e.g., username).

        Returns:
        dict: A dictionary with emotion names as keys and their counts as values.
        """
        # Filter the dataframe to get rows where the author_id matches
        author_data = self.df[self.df['author'] == author_id]

        # Sum up the emotions for this author
        emotion_distribution = author_data[self.emotion_columns].sum(axis=0)

        return emotion_distribution.to_dict()


    def get_authors_with_emotion(self, emotion):
        """
        Returns the authors who have a specific emotion.

        Parameters:
        emotion (str): The emotion to filter by (e.g., 'joy').

        Returns:
        list: A list of authors (author_ids) who have the specified emotion.
        """
        # Check if the emotion exists in the columns
        if emotion not in self.emotion_columns:
            raise ValueError(f"Emotion '{emotion}' not found in the dataframe.")

        # Filter authors who have the emotion (value > 0)
        authors_with_emotion = self.df[self.df[emotion] > 0]['author'].unique()

        return authors_with_emotion.tolist()



    def get_average_emotion_for_author(self, author_id):
        """
        Returns the average emotion score for a specific author.

        Parameters:
        author_id (str): The ID of the author (e.g., username).

        Returns:
        dict: A dictionary with emotion names as keys and their average scores as values.
        """
        # Filter the dataframe to get rows where the author_id matches
        author_data = self.df[self.df['author'] == author_id]

        # Calculate the average score for each emotion
        average_emotion = author_data[self.emotion_columns].mean(axis=0)

        return average_emotion.to_dict()

    def get_emotion_correlation_for_author(self, author_id):
        """
        Returns the correlation matrix for the emotions of a specific author.

        Parameters:
        author_id (str): The ID of the author (e.g., username).

        Returns:
        pd.DataFrame: A correlation matrix of emotions for the specified author.
        """
        # Filter the dataframe to get rows where the author_id matches
        author_data = self.df[self.df['author'] == author_id]

        # Calculate the correlation between emotions for this author
        emotion_correlation = author_data[self.emotion_columns].corr()

        return emotion_correlation

    def get_total_emotions_and_appearances(self, author_name):
        """Calculate total emotions and appearances for the author."""
        # Check if the author exists
        if author_name not in self.author_emotions.index:
            print(f"Author '{author_name}' not found.")
            return None, None

        # Get the author's data
        author_data = self.author_emotions.loc[author_name]

        # If the author's data is empty or contains only null values
        if author_data.isnull().all():
            print(f"No emotion data available for author '{author_name}'.")
            return None, None

        # Calculate total emotions
        total_emotions = author_data[self.emotion_columns].sum()

        # Calculate the number of appearances for the author
        appearances = self.df[self.df['author'] == author_name].shape[0]

        return total_emotions, appearances

    def get_exceeding_emotions(self, author_data):
        """Identify emotions exceeding the overall average."""
        exceeding_emotions = author_data[author_data > self.overall_emotions_avg]
        return exceeding_emotions

    def calculate_author_percentage(self, author_data):
        """Calculate the percentage contribution of the author to the total emotions."""
        total_emotions_all_authors = self.author_emotions.sum().sum()
        author_total_emotions = author_data.sum()
        percentage = (author_total_emotions / total_emotions_all_authors) * 100
        return percentage

    def plot_emotion_distribution(self, author_data, author_name):
        """Plot a bar chart of the author's emotions."""
        sorted_emotions = author_data.sort_values(ascending=False)
        plt.figure(figsize=(10, 6))
        sorted_emotions.plot(kind='bar', color='skyblue', alpha=0.8)
        plt.title(f"Emotion Distribution for Author '{author_name}'", fontsize=14)
        plt.xlabel("Emotions", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def plot_emotion_radar_chart(self, author_data, author_name):
        """Plot a radar chart of the author's emotions."""
        categories = author_data.index
        values = author_data.values
        N = len(categories)

        angles = [n / float(N) * 2 * pi for n in range(N)]
        values = np.append(values, values[0])
        angles += angles[:1]

        plt.figure(figsize=(8, 8))
        ax = plt.subplot(111, polar=True)
        plt.xticks(angles[:-1], categories, color='grey', size=12)
        ax.plot(angles, values, linewidth=2, linestyle='solid', color='blue')
        ax.fill(angles, values, color='skyblue', alpha=0.4)
        plt.title(f"Emotion Radar Chart for Author '{author_name}'", size=15)
        plt.show()

    def display_author_emotion_summary(self, author_name, total_emotions, appearance_count, exceeding_emotions, author_percentage):
        """Display a summary of the author's emotions."""
        print(f"Total emotions for author '{author_name}': {total_emotions}")
        print(f"Author '{author_name}' appears {appearance_count} times in the dataset.")
        print(f"Number of emotions exceeding the overall average: {len(exceeding_emotions)}")

        if not exceeding_emotions.empty:
            print("\nEmotions exceeding the overall average:")
            for emotion, value in exceeding_emotions.items():
                print(f"  {emotion}: {value}")
        else:
            print("\nNo emotions exceed the overall average.")

        print(f"\nAuthor '{author_name}' contributes {author_percentage:.2f}% to the total emotions in the dataset.")


    def analyze_author1(self, author_name):
        """Main function to analyze author emotions."""
        # print(self.author_emotions.loc[self.author_emotions.index == author_name])
        if author_name in self.author_emotions.index.tolist():
            author_data = self.author_emotions.loc[author_name]

            # Step 1: Total emotions and appearances
            total_emotions, appearance_count = self.get_total_emotions_and_appearances(author_name)

            # Step 2: Exceeding emotions
            exceeding_emotions = self.get_exceeding_emotions(author_data)

            # Step 3: Author percentage contribution
            author_percentage = self.calculate_author_percentage(author_data)

            # Step 4: Display summary
            self.display_author_emotion_summary(author_name, total_emotions, appearance_count, exceeding_emotions, author_percentage)

            # Step 5: Plot distributions
            self.plot_emotion_distribution(author_data, author_name)
            self.plot_emotion_radar_chart(author_data, author_name)

            return author_data.sort_values(ascending=False)
        else:
            return f"Author '{author_name}' not found in the data."

    def analyze_user_emotions(self, target_user):
        """Calculate and plot various metrics for a user's emotions."""
        combined_df = self.df.copy()  # Assuming this is your main dataframe with user and date columns
        combined_df['date'] = combined_df['created_utc'].dt.date  # Assuming 'created_utc' is a datetime column
        user_emotions1 = combined_df.groupby(['author', 'date'])[self.emotion_columns].sum()

        # Aggregating emotions by date for trend analysis
        user_data = user_emotions1.loc[target_user]
        user_data_by_date = user_data.groupby('date')[self.emotion_columns].sum()

        # 1. Plotting emotion trends over time
        plt.figure(figsize=(12, 6))
        for emotion in self.emotion_columns:
            plt.plot(user_data_by_date.index, user_data_by_date[emotion], label=emotion)

        plt.title(f"Emotion Trends Over Time for User '{target_user}'", fontsize=14)
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Emotion Count", fontsize=12)
        plt.legend(title="Emotions", fontsize=10)
        plt.grid(alpha=0.4)
        plt.tight_layout()
        plt.show()

        # 2. Comparing the user's emotions to other users
        overall_avg = self.author_emotions[self.emotion_columns].mean()
        user_avg = user_data[self.emotion_columns].mean()
        comparison = pd.DataFrame({'Overall Average': overall_avg, f"{target_user} Average": user_avg})

        # Bar chart for comparison
        comparison.plot(kind='bar', figsize=(10, 6), color=['gray', 'skyblue'], alpha=0.8)
        plt.title(f"Comparison of {target_user}'s Average Emotions vs Overall", fontsize=14)
        plt.xlabel("Emotions", fontsize=12)
        plt.ylabel("Average Count", fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(alpha=0.4)
        plt.tight_layout()
        plt.show()

        # 3. Emotion intensity metric (average per interaction)
        user_interactions = len(user_data)
        intensity_metric = user_data[self.emotion_columns].sum().sum() / user_interactions
        print(f"Emotion Intensity Metric for {target_user}: {intensity_metric:.2f}")

        # 4. Correlation between emotions for the user
        correlation_matrix = user_data[self.emotion_columns].corr()

        # Heatmap for correlations
        plt.figure(figsize=(24, 12))
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", cbar=True)
        plt.title(f"Emotion Correlations for User '{target_user}'", fontsize=14)
        plt.tight_layout()
        plt.show()

        # 5. Radar chart for user emotion distribution
        user_avg_sorted = user_avg.sort_values()
        categories = user_avg_sorted.index.tolist()
        values = user_avg_sorted.values.tolist()
        values += values[:1]
        angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))]
        angles += angles[:1]

        plt.figure(figsize=(8, 8))
        ax = plt.subplot(111, polar=True)
        plt.xticks(angles[:-1], categories, color='black', fontsize=12)
        ax.plot(angles, values, linewidth=2, linestyle='solid', color='blue')
        ax.fill(angles, values, color='skyblue', alpha=0.4)
        plt.title(f"Emotion Radar Chart for User '{target_user}'", size=15)
        plt.tight_layout()
        plt.show()

        return {
            'emotion_trends': user_data_by_date,
            'comparison': comparison,
            'intensity_metric': intensity_metric,
            'correlation_matrix': correlation_matrix,
            'user_avg': user_avg
        }

    def count_raters_above_emotion_threshold(self, emotions=None, percent=50):
        """Count the raters who have emotion percentage above the threshold.
        Accepts a list of emotions or a single emotion."""

        # If emotions are provided, validate if they exist in the columns
        if emotions:
            if isinstance(emotions, str):  # If a single emotion is provided
                emotions = [emotions]
            # Check if all provided emotions exist in the columns
            for emotion in emotions:
                if emotion not in self.author_emotions.columns:
                    return f"Emotion '{emotion}' not found in the data."

        total_tags_per_rater = self.author_emotions.sum(axis=1)

        result = {}

        if emotions:
            # If specific emotions are provided, process each one
            for emotion in emotions:
                # Calculate emotion percentage per rater
                emotion_percent_per_rater = (self.author_emotions[emotion] / total_tags_per_rater) * 100
                raters_above_threshold = emotion_percent_per_rater[emotion_percent_per_rater >= percent]
                result[emotion] = {
                    "raters_above_threshold": list(raters_above_threshold.index),
                    "count": len(raters_above_threshold)
                }
        else:
            # If no specific emotions are provided, analyze for all emotions
            for emotion in self.author_emotions.columns:
                # Calculate emotion percentage per rater
                emotion_percent_per_rater = (self.author_emotions[emotion] / total_tags_per_rater) * 100
                raters_above_threshold = emotion_percent_per_rater[emotion_percent_per_rater >= percent]
                result[emotion] = {
                    "raters_above_threshold": list(raters_above_threshold.index),
                    "count": len(raters_above_threshold)
                }

        return result
    def plot_emotion_percentages(self):
        """Plot the percentage of users tagging each emotion."""
        total_users = len(self.author_emotions)
        emotion_percentages = (self.author_emotions.astype(bool).sum(axis=0) / total_users) * 100

        # Sort emotions by percentage in descending order
        sorted_emotions = emotion_percentages.sort_values(ascending=False)

        # Create a bar plot for emotion percentages
        plt.figure(figsize=(10, 6))
        sorted_emotions.plot(kind='bar', color='skyblue', alpha=0.8)
        plt.title("Percentage of Users Tagging Each Emotion", fontsize=14)
        plt.xlabel("Emotions", fontsize=12)
        plt.ylabel("Percentage of Users (%)", fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(alpha=0.4)
        plt.tight_layout()
        plt.show()

    def plot_emotions_by_thresholds(self, thresholds=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100], emotions=None):
        """Plot emotions by the number of raters above each threshold."""
        if emotions is None:
            emotions = self.author_emotions.columns  # Use all emotions if no specific emotions are provided

        emotion_threshold_data = []

        for emotion in emotions:
            emotion_counts = []
            print(emotion)
            for threshold in thresholds:
                print(emotion,threshold)
                result = self.count_raters_above_emotion_threshold(emotions=emotion, percent=threshold)
                print(result[emotion]['count'])
                emotion_counts.append(result[emotion]['count'])

            emotion_threshold_data.append(emotion_counts)

            # Creating bar graph for each emotion
            fig, ax = plt.subplots(figsize=(10, 6))

            ax.bar(thresholds, emotion_counts, width=8, color='skyblue')
            ax.set_title(f"Number of Raters Above {emotion} Emotion Thresholds", fontsize=14)
            ax.set_xlabel("Threshold (%)", fontsize=12)
            ax.set_ylabel("Number of Raters", fontsize=12)
            ax.set_xticks(thresholds)
            ax.set_xticklabels(thresholds)
            ax.grid(True, alpha=0.4)
            plt.tight_layout()
            plt.show()

        emotion_threshold_matrix = pd.DataFrame(emotion_threshold_data, columns=thresholds, index=emotions)

        return emotion_threshold_matrix

    def get_responses_by_user(self, selected_user):
        """Return the rows where the specified user has responded."""
        responded_to_user = self.df[self.df['author'] == selected_user]
        return responded_to_user

    def count_reactions_and_responses(self, selected_user):
        """Count the reactions and responses to the selected user, and extract emotions."""
        # Filter for posts made by the selected user
        responded_to_user = self.df[self.df['author'] == selected_user]
        num_responses = responded_to_user.shape[0]  # Number of posts by the selected user

        # Filter for comments made by others in response to the selected user
        user_responses = self.df[self.df['parent_id'].isin(responded_to_user['link_id'])]
        num_reactions = user_responses.shape[0]  # Number of comments received by the selected user

        # Extracting the authors of the responses (those who commented on the selected user)
        responding_authors = user_responses['author'].unique().tolist()

        # Extracting the posts that the selected user has commented on, ensuring no duplicates
        commented_posts = self.df[
            self.df['parent_id'].isin(user_responses['link_id'])].drop_duplicates(subset='link_id')
        commented_posts_data = []
        for _, row in commented_posts.iterrows():
            emotions = row[self.emotion_columns].to_dict()  # Get emotions for each commented post

            # Filter emotions with value 1
            emotions_with_1 = {emotion: value for emotion, value in emotions.items() if value == 1}

            # Store only emotions with value 1 for the commented post
            commented_data = {
                'commented_post_id': row['link_id'],
                'commenter': row['author'],
                'emotions': emotions_with_1  # Store emotions with value 1
            }
            commented_posts_data.append(commented_data)

        # Adding emotions for each response to the selected user
        responses_with_emotions = []
        for _, row in user_responses.iterrows():
            emotions = row[self.emotion_columns].to_dict()  # Get emotions for each response

            # Filter emotions with value 1
            emotions_with_1 = {emotion: value for emotion, value in emotions.items() if value == 1}

            # Store only emotions with value 1 for the response
            response_data = {
                'response_id': row['link_id'],
                'responder': row['author'],
                'emotions': emotions_with_1  # Store emotions with value 1
            }
            responses_with_emotions.append(response_data)

        return {
            'user': selected_user,
            'num_responses': num_responses,
            'num_reactions': num_reactions,
            'responding_authors': responding_authors,
            'responses_with_emotions': responses_with_emotions,
            'commented_posts_data': commented_posts_data  # List of posts the selected user commented on
        }




def main():
    # Load the dataframe (replace with actual method to load data)
    data_path = 'data/full_dataset/goemotions_combined.csv'
    df = pd.read_csv(data_path, encoding='utf-8')

    # Create an instance of the AuthorEmotionAnalysis class
    author_analysis = AuthorEmotionAnalysis(df)
    selected_user_reactions = author_analysis.count_reactions_and_responses("saturdeity")
    print(selected_user_reactions)
    # emotion = 'joy'  # Replace with the emotion you want to analyze
    # thresholds = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    # emotion_threshold_matrix = author_analysis.plot_emotions_by_thresholds(thresholds=thresholds)
    #
    # # Display the matrix of emotions vs thresholds
    # print(emotion_threshold_matrix)
    # percent = 100  # Set your desired threshold percentage
    #
    # # Call the function to get raters above the threshold
    # result = author_analysis.count_raters_above_emotion_threshold(emotions=None, percent=100)
    #
    # # Check if the result is a dictionary (it should be in your case)
    # if isinstance(result, dict):
    #     for emotion, data in result.items():
    #         print(f"Number of raters tagging '{emotion}' above {percent}%: {data['count']}")
    #         print(f"Users: {data['raters_above_threshold']}")
    # else:
    #     print(result)  # Handle case where result is an error message or non-dict
    # Print the result
    # print(f"Raters who have more than {percent}% of  emotion:")
    # print(raters_above_threshold)

    # target_user = '[deleted]'  # Replace this with an actual author name from your dataset
    #
    # # Analyze emotions for the target user
    # analysis_results = author_analysis.analyze_user_emotions(target_user)
    #
    # # Print out the results
    # print("Emotion Trends:")
    # print(analysis_results['emotion_trends'])
    #
    # print("\nComparison of User's Average Emotions vs Overall:")
    # print(analysis_results['comparison'])
    #
    # print(f"\nEmotion Intensity Metric for {target_user}: {analysis_results['intensity_metric']:.2f}")
    #
    # print("\nCorrelation Matrix:")
    # print(analysis_results['correlation_matrix'])
    #
    # print("\nUser's Average Emotions:")
    # print(analysis_results['user_avg'])
    #
    # # author = input("Enter the name of the author to analyze: ")
    #
    # # Analyze the specified author
    # author_id = 'CakeDay--Bot'
    # result = author_analysis.analyze_author1(author_id)
    #
    # if isinstance(result, str):
    #     print(result)  # Print the error message if the author was not found.
    # else:
    #     print("\nAnalysis completed successfully.")
    #     print(f"\nTop emotions for '{author_id}':")
    #     print(result.head())
    # Example usage:
    # author_id = 'CakeDay--Bot'
    #
    # # Get emotion distribution for a specific author
    # emotion_distribution = author_analysis.get_emotion_distribution_for_author(author_id)
    # print(f"Emotion distribution for author {author_id}: {emotion_distribution}")

    # Get authors with a specific emotion
    # authors_with_joy = author_analysis.get_authors_with_emotion('joy')
    # print(f"Authors with Joy emotion: {authors_with_joy}")
    #
    # # Get average emotion score for a specific author
    # average_emotion = author_analysis.get_average_emotion_for_author(author_id)
    # print(f"Average emotion scores for {author_id}: {average_emotion}")
    #
    # # Get emotion correlation for a specific author
    # emotion_correlation = author_analysis.get_emotion_correlation_for_author(author_id)
    # print(f"Emotion correlation for {author_id}:")
    # print(emotion_correlation)


# Run the main function if this script is executed
if __name__ == "__main__":
    main()
