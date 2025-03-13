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



class SubredditEmotionAnalysis:
    def __init__(self, combined_df):
        self.combined_df = combined_df
        self.combined_df['created_utc'] = pd.to_datetime(self.combined_df['created_utc'], errors='coerce')
        self.emotion_columns = combined_df.columns[9:]
        self.subreddit_emotions = combined_df.groupby('subreddit', as_index=False)[self.emotion_columns].sum()
        self.subreddit_dfs = {}
        self.create_subreddit_dfs()
        self.status_dict = defaultdict(lambda: {'comments': {}, 'subreddits': set()})
        self.build_comment_status_dict()

    def create_subreddit_dfs(self):
        for subreddit, group in self.combined_df.groupby('subreddit'):
            self.subreddit_dfs[subreddit] = group.reset_index(drop=True)

    def get_sub_matr(self):
        return self.subreddit_emotions

    def get_sub(self):
        return self.combined_df['subreddit'].unique()

    def analyze_subreddit_emotions(self):
        subreddit_emotion_analysis = {}
        for subreddit, subreddit_df in self.subreddit_dfs.items():
            num_records = len(subreddit_df)
            print(f"Data for subreddit '{subreddit}': with {num_records} records:")
            emotion_counts = {
                emotion: (subreddit_df[emotion] == 1).sum()
                for emotion in self.emotion_columns
            }
            emotion_percentages = {
                emotion: (count / num_records) * 100 if num_records > 0 else 0
                for emotion, count in emotion_counts.items()
            }
            subreddit_emotion_analysis[subreddit] = {
                'emotion_counts': emotion_counts,
                'emotion_percentages': emotion_percentages,
            }

        return subreddit_emotion_analysis

    def analyze_subreddit_stats(self, subreddit_name, date_column='created_utc'):
        """
        Analyze subreddit statistics, including average emotions and comments by date.

        Args:
            subreddit_name (str): The name of the subreddit to analyze.
            date_column (str): The name of the column containing datetime information. Default is 'created_utc'.

        Returns:
            None. Displays heatmap and bar plot for the analysis.
        """
        # Filter data for the specified subreddit
        subreddit_data = self.combined_df[self.combined_df['subreddit'] == subreddit_name]
        if subreddit_data.empty:
            print(f"No data found for subreddit: {subreddit_name}")
            return

        # Create a copy to avoid modifying the original DataFrame
        subreddit_data = subreddit_data.copy()

        # Check if the specified date column exists in the data
        if date_column not in subreddit_data.columns:
            print(f"Column '{date_column}' not found in the dataset.")
            return

        # Convert the date column to datetime format, coercing invalid entries to NaT
        subreddit_data[date_column] = pd.to_datetime(subreddit_data[date_column], errors='coerce')

        # Check if all entries in the date column are invalid
        if subreddit_data[date_column].isnull().all():
            print(f"All values in '{date_column}' are invalid or could not be converted.")
            return

        # Extract just the date from the datetime column and create a new column
        subreddit_data['created_date'] = subreddit_data[date_column].dt.date

        # Calculate mean values for the emotion columns
        emotion_means = subreddit_data[self.emotion_columns].mean()

        # Count comments by date, sorted in chronological order
        comments_by_date = subreddit_data['created_date'].value_counts().sort_index()

        # Display the average emotions
        print(f"Average emotions for subreddit '{subreddit_name}':\n{emotion_means}")

        # Generate a heatmap to visualize the average emotions
        plt.figure(figsize=(16, 10))
        sns.heatmap(emotion_means.to_frame().T, cmap="coolwarm", annot=True, cbar=False)
        plt.title(f"Average Emotions for Subreddit: {subreddit_name}")
        plt.xlabel("Emotion")
        plt.ylabel("Value")
        plt.show()

        # Generate a bar plot for comments per date
        plt.figure(figsize=(16, 6))
        comments_by_date.plot(kind='bar', color='skyblue')
        plt.title(f"Number of Comments Per Date for Subreddit: {subreddit_name}")
        plt.xlabel("Date")
        plt.ylabel("Number of Comments")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        # Calculate and print the top 5 most frequent emotions
        emotion_counts = subreddit_data[self.emotion_columns].sum().sort_values(ascending=False)
        print(f"Top 5 most frequent emotions for subreddit '{subreddit_name}':\n{emotion_counts.head()}\n")

        # Plot the top 5 most frequent emotions as a bar chart
        plt.figure(figsize=(10, 6))
        emotion_counts.head().plot(kind='bar', color='skyblue', edgecolor='black')
        plt.title(f"Top 5 Most Frequent Emotions for Subreddit: {subreddit_name}")
        plt.xlabel("Emotion Type")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.show()

        # Plot correlation heatmap for emotions
        emotion_corr = subreddit_data[self.emotion_columns].corr()
        plt.figure(figsize=(20, 12))
        sns.heatmap(emotion_corr, annot=True, cmap="coolwarm", fmt=".2f", cbar=True)
        plt.title(f"Correlation of Emotions for Subreddit: {subreddit_name}")
        plt.tight_layout()
        plt.show()
    def build_network_graph(self, subreddit_name):
        subreddit_data = self.combined_df[self.combined_df['subreddit'] == subreddit_name]
        if subreddit_data.empty:
            print(f"No data found for subreddit: {subreddit_name}")
            return
        G = nx.DiGraph()
        emotion_colors = {
            'admiration': 'lightblue', 'amusement': 'yellow', 'anger': 'red',
            'annoyance': 'orange', 'approval': 'green', 'caring': 'pink',
            'confusion': 'purple', 'curiosity': 'teal', 'desire': 'brown',
            'disappointment': 'gray', 'disapproval': 'darkgreen', 'disgust': 'darkviolet',
            'embarrassment': 'lightgreen', 'excitement': 'cyan', 'fear': 'indigo',
            'gratitude': 'lightcoral', 'grief': 'darkgray', 'joy': 'gold',
            'love': 'hotpink', 'nervousness': 'darkorange', 'optimism': 'skyblue',
            'pride': 'darkred', 'realization': 'limegreen', 'relief': 'mediumblue',
            'remorse': 'chocolate', 'sadness': 'blue', 'surprise': 'pink',
            'neutral': 'lightgray'
        }

        for index, comment in subreddit_data.iterrows():
            comment_id = comment['id']
            author = comment['author']
            text = comment['text']

            # Get emotion data for the comment
            comment_emotions = comment[self.emotion_columns]  # This will get all the emotion columns for the comment

            # Get the dominant emotion (the one with the highest score)
            dominant_emotion = comment_emotions.idxmax()  # Get the emotion with the highest value

            # Add node with emotion data
            G.add_node(comment_id, author=author, text=text, emotions=dominant_emotion)

            # If the comment has a parent (is a reply), add an edge to the parent
            parent_id = comment['parent_id']
            if pd.notnull(parent_id) and parent_id != comment_id:
                # Assign the color based on the dominant emotion of the comment
                edge_color = emotion_colors.get(dominant_emotion, 'gray')  # Default to gray if no match
                G.add_edge(parent_id, comment_id, emotion=dominant_emotion, color=edge_color)

            # Plotting the graph
        plt.figure(figsize=(24, 14))
        pos = nx.spring_layout(G, seed=42)  # Layout for positioning nodes

        # Draw nodes with color based on dominant emotion
        node_colors = []
        for node in G.nodes():
            emotion = G.nodes[node].get('emotions', 'neutral')  # Get the emotion or use 'neutral' as fallback
            node_colors.append(emotion_colors.get(emotion, 'lightgray'))  # Default to 'lightgray' if emotion not found

        nx.draw_networkx_nodes(G, pos, node_size=500, node_color=node_colors, alpha=0.7)

        # Draw edges with color based on the emotion
        edge_colors = [G[u][v].get('color', 'lightgray') for u, v in G.edges()]
        nx.draw_networkx_edges(G, pos, edge_color=edge_colors, alpha=0.5, width=1.5)

        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=8, font_color='black')

        # Add a color legend
        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=emotion)
                   for emotion, color in emotion_colors.items()]
        plt.legend(handles=handles, title="Emotions", loc="best", fontsize=12)

        # Display the graph title
        plt.title(f"Network Graph for Subreddit: {subreddit_name} with Emotions")
        plt.tight_layout()
        plt.show()

    def get_unique_words_for_topic(self, subreddit_name, stop_words=ENGLISH_STOP_WORDS):
        """
        Get the most common unique words for a given subreddit, excluding stopwords.

        Args:
            subreddit_name (str): The name of the subreddit to analyze.
            stop_words (set): A set of stopwords to exclude from the analysis (default: ENGLISH_STOP_WORDS).

        Returns:
            list: A list of tuples representing the most common words and their counts.
        """
        # Filter the DataFrame for the specified subreddit
        subreddit_data = self.combined_df[self.combined_df['subreddit'] == subreddit_name]

        if subreddit_data.empty:
            print(f"No data found for subreddit: {subreddit_name}")
            return []

        # Combine all text data into a single string, dropping NaN values
        all_text = " ".join(subreddit_data['text'].dropna())

        # Extract English words using regex
        english_words = re.findall(r'\b[a-zA-Z]+\b', all_text)

        # Convert words to lowercase and filter out stopwords
        filtered_words = [word.lower() for word in english_words if word.lower() not in stop_words]

        # Count the frequency of each word
        word_counts = Counter(filtered_words)

        # Return the top 10 most common words with their counts
        return word_counts.most_common(10)


    def build_comment_status_dict(self):
        # Loop over all rows in the dataframe
        for idx, row in self.combined_df.iterrows():
            comment_id = row['id']
            parent_id = row['parent_id']
            link_id = row['link_id']
            comment_author = row['author']
            comment_text = row['text']
            subreddit = row['subreddit']  # Assuming 'subreddit' column exists in your dataframe
            created_utc = row['created_utc']  # The UTC time of the comment

            # If parent_id is NaN, use link_id as the status key
            status_key = parent_id if pd.notnull(parent_id) else link_id

            # Add the comment to the status dictionary (ensure no duplicates by checking the comment_id)
            if comment_id not in self.status_dict[status_key]['comments']:
                self.status_dict[status_key]['comments'][comment_id] = {
                    'author': comment_author,
                    'text': comment_text,
                    'created_utc': created_utc
                }

            # Add the subreddit to the set for that status
            self.status_dict[status_key]['subreddits'].add(subreddit)

    def count_comments_by_status(self):
        # Count the number of comments for each status and get the associated subreddits
        comment_counts = {}
        for status_key, data in self.status_dict.items():
            comment_counts[status_key] = {
                'count': len(data['comments']),
                'subreddits': list(data['subreddits'])  # List of unique subreddits for each status
            }

        return comment_counts

def main():
    data_path = 'data/full_dataset/goemotions_combined.csv'
    df = pd.read_csv(data_path, encoding='utf-8')



    # Initialize the SubredditEmotionAnalysis class
    analysis = SubredditEmotionAnalysis(df)
    print(analysis.get_sub_matr())
    # Example subreddit name to analyze
    print(analysis.get_sub())
    subreddit_name = 'politics'  # Replace with your desired subreddit

    # # Perform emotion analysis for all subreddits
    print("\nAnalyzing emotions for all subreddits...\n")
    all_subreddit_emotion_analysis = analysis.analyze_subreddit_stats(subreddit_name)

    # Perform detailed analysis for a specific subreddit
    print(f"\nPerforming detailed analysis for subreddit: {subreddit_name}\n")
    analysis.analyze_subreddit_stats(subreddit_name)

    # Build and visualize network graph for a specific subreddit
    print(f"\nBuilding network graph for subreddit: {subreddit_name}\n")
    analysis.build_network_graph(subreddit_name)

    # Additional example for extracting unique words for a subreddit
    print(f"\nExtracting unique words for subreddit: {subreddit_name}\n")
    unique_words = analysis.get_unique_words_for_topic(subreddit_name)
    print(f"Top unique words in subreddit '{subreddit_name}':\n{unique_words}")
    status_counts = analysis.count_comments_by_status()
    print(status_counts)


if __name__ == "__main__":
    main()