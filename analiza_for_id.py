import pandas as pd
import matplotlib
# matplotlib.use('TkAgg')
class EmotionAnalysis:
    def __init__(self, combined_df):
        """
        Initializes the EmotionAnalysis class with the given DataFrame.

        Parameters:
        combined_df (DataFrame): The DataFrame containing the data.
        """
        self.combined_df = combined_df

        # Identify columns that represent emotions (assuming they start with 'id')
        self.emotion_columns = combined_df.columns[9:]

    def get_ids_with_emotion(self, emotion_column, emotion_value=1):
        """
        Returns all 'id's that have a certain emotion value (default is 1) in the given emotion column.

        Parameters:
        emotion_column (str): The emotion column to check.
        emotion_value (int): The value of the emotion to filter by (default is 1 for present emotion).

        Returns:
        DataFrame: A DataFrame with ids and their corresponding emotion counts.
        """
        filtered_df = self.combined_df[self.combined_df[emotion_column] == emotion_value]

        emotion_count_df = filtered_df.groupby('id').size().reset_index(name=f'{emotion_column}_count')

        return emotion_count_df

    def get_ids_with_multiple_emotions(self):
        """
        Returns a dataframe with the ids that have multiple emotions.

        Returns:
        DataFrame: A DataFrame with ids and their multiple emotions.
        """
        emotion_data = self.combined_df[['id'] + list(self.emotion_columns)]
        emotion_data = emotion_data.copy()
        # Calculate the sum of emotions for each id
        emotion_data.loc[:, 'emotion_count'] = emotion_data[self.emotion_columns].sum(axis=1)


        # Filter ids with more than one emotion
        multiple_emotion_data = emotion_data[emotion_data['emotion_count'] > 1]

        emotion_counts = multiple_emotion_data.groupby('id')[self.emotion_columns].sum()

        return emotion_counts

    def get_emotion_id_dict_for_unique(self):
        """
        Generates a dictionary where each emotion is a key, and the value is a list of ids
        that appear only once in the dataset and are associated with that emotion.

        Returns:
        dict: A dictionary with emotions as keys and lists of ids as values.
        """
        id_counts = self.combined_df['id'].value_counts()

        # Filter for ids that appear only once
        unique_id_data = self.combined_df[self.combined_df['id'].isin(id_counts[id_counts == 1].index)]

        emotion_id_dict = {emotion: [] for emotion in self.emotion_columns}

        for idx, row in unique_id_data.iterrows():
            for emotion in self.emotion_columns:
                if row[emotion] == 1:
                    emotion_id_dict[emotion].append(row['id'])

        return emotion_id_dict

    def emotion_distribution_for_id(self, user_id):
        """
        Calculates the emotion distribution for a given user ID.

        Parameters:
        user_id (str): The id of the user to get the emotion distribution for.

        Returns:
        Series: The sum of each emotion for that user id, or a message if the id is not found.
        """
        user_data = self.combined_df[self.combined_df['id'] == user_id]

        if user_data.empty:
            return f"ID {user_id} not found in the dataset."

        emotion_distribution = user_data[self.emotion_columns].sum(axis=0)

        return emotion_distribution

    def emotion_correlations(self):
        """
        Calculates and returns the correlation between emotion columns and visualizes it.

        Returns:
        DataFrame: The correlation matrix of the emotions.
        """
        # Compute the correlation matrix
        correlation_matrix = self.combined_df[self.emotion_columns].corr()

        import seaborn as sns
        import matplotlib.pyplot as plt

        # Create a heatmap for the correlation matrix
        plt.figure(figsize=(24, 12))  # Adjust size as needed
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f', linewidths=0.5)
        plt.title('Emotion Correlation Heatmap')
        plt.show()

        return correlation_matrix



def main():
    # Load your DataFrame (replace this with your actual DataFrame loading method)
    data_path = 'data/full_dataset/goemotions_combined.csv'
    df = pd.read_csv(data_path)

    # Create an instance of the EmotionAnalysis class
    emotion_analysis = EmotionAnalysis(df)

    # Example: Get IDs with a certain emotion
    # ids_with_joy = emotion_analysis.get_ids_with_emotion('joy')
    # print("IDs with Joy Emotion:", ids_with_joy)

    # Example: Get IDs with multiple emotions
    # multiple_emotions_count = emotion_analysis.get_ids_with_multiple_emotions()
    # print("Multiple Emotions Count:", multiple_emotions_count)

    # # Example: Get the emotion dictionary for unique IDs
    # emotion_id_dict = emotion_analysis.get_emotion_id_dict_for_unique()
    # print("Emotion ID Dictionary:", emotion_id_dict)

    # # Example: Get emotion distribution for a specific user ID
    # user_id = 'eczbdg4'
    # emotion_distribution = emotion_analysis.emotion_distribution_for_id(user_id)
    # print(f"Emotion Distribution for ID {user_id}:", emotion_distribution)
    #
    # # Example: Get the emotion correlations
    correlation_matrix = emotion_analysis.emotion_correlations()
    print("Emotion Correlation Matrix:", correlation_matrix)


# Run the main function if this script is executed
if __name__ == "__main__":
    main()