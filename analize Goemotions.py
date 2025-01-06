import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

# Set backend for Matplotlib to avoid errors
# matplotlib.use('TkAgg')

class DataAnalysis:
    def __init__(self, dataframe):
        # Initialize with the given dataframe
        self.df = dataframe
        self.emotion_columns =dataframe.columns[9:]

    def get_columns(self):
        return list(self.df.columns)
    # Display a general summary of the data
    def summary(self):
        print("General summary of the data:")
        print(self.df.info())

    # Display information about missing values in the dataframe
    def missing_values(self):
        missing_data = self.df.isnull().sum()
        missing_data = missing_data[missing_data > 0]
        if not missing_data.empty:
            print("\nMissing values in the following columns:")
            print(missing_data)
        else:
            print("\nNo missing values found.")

    def unique_values(self):
        # Count unique values for each column
        unique_counts = self.df.nunique()

        # Print the unique counts
        print("\nUnique counts for each column:")
        print(unique_counts)

    def emotion_counts(self):
        # Sum the values in the emotion columns
        emotion_counts = self.df[self.emotion_columns].sum()

        # Print the emotion counts
        print("\nEmotion counts for each emotion category:")
        print(emotion_counts)
    # Plot histograms for all numeric columns in the dataframe
    # def plot_histograms(self):
    #     # Select numeric columns
    #     numeric_columns = self.df.select_dtypes(include=['float64', 'int64']).columns
    #     print("\nHistograms for numeric columns:")
    #     # Plot histograms
    #     self.df[numeric_columns].hist(figsize=(10, 8), bins=30)
    #     plt.show()
    #
    # # Compute and display a correlation heatmap for the numeric columns
    # def correlation_heatmap(self):
    #     # Calculate correlation matrix
    #     correlation_matrix = self.df.corr()
    #     print("\nCorrelation between columns:")
    #     # Display heatmap
    #     sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    #     plt.show()


# Example usage:
if __name__ == '__main__':
    # Read the combined dataset (assuming you have already downloaded it)
    data_path = 'data/full_dataset/goemotions_combined.csv'
    df = pd.read_csv(data_path)

    # Create an instance of DataAnalysis and perform various analyses

    analysis = DataAnalysis(df)
    print(analysis.get_columns())
    analysis.summary()  # Display summary statistics
    analysis.missing_values()  # Check for missing values
    analysis.unique_values()
    analysis.emotion_counts()
    # analysis.plot_histograms()  # Plot histograms for numeric columns
    # analysis.correlation_heatmap()  # Display correlation heatmap
