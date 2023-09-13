import pandas as pd
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk

# Download stopwords if not already downloaded
nltk.download('stopwords')

def preprocess_data(dataframe):
    # Preprocessing code here
    stop_words = set(stopwords.words('english'))
    dataframe['clean_text'] = dataframe['Summary'].apply(lambda x: ' '.join([word for word in str(x).lower().split() if word not in stop_words]))
    dataframe['clean_text'] = dataframe['clean_text'].apply(lambda x: re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', '', x))
    dataframe['clean_text'] = dataframe['clean_text'].apply(lambda x: re.sub(r'http\S+|www\S+', '', x))
    dataframe['clean_text'] = dataframe['clean_text'].apply(lambda x: re.sub(r'<.*?>', '', x))
    return dataframe

def calculate_similarity(df, threshold):
    # Initialize the TF-IDF vectorizer
    vectorizer = TfidfVectorizer()

    # Compute the TF-IDF matrix for the summaries
    tfidf_matrix = vectorizer.fit_transform(df['clean_text'])

    # Compute the cosine similarity matrix
    similarity_matrix = cosine_similarity(tfidf_matrix)

    # Get the indices of summaries in ascending order of similarity from summary 0
    summary_index = 0

    # Create a list to store the pairs of nodes and their similarities
    pairs = []

    # Iterate over the similarity matrix
    for i in range(len(similarity_matrix)):
        for j in range(i + 1, len(similarity_matrix)):
            if similarity_matrix[i, j] > threshold:  # Set a threshold for similarity
                summary1_id = df['Issue key'].iloc[i]
                summary2_id = df['Issue key'].iloc[j]
                similarity = similarity_matrix[i, j]

                # Append the pair of nodes and their similarity to the list
                pairs.append((summary1_id, summary2_id, similarity))

    # Convert the list of pairs to a DataFrame
    pairs_df = pd.DataFrame(pairs, columns=['Node 1', 'Node 2', 'Similarity'])

    # Merge with original DataFrame to include Issue key, Summary, and Story Points columns
    merged_df = pd.merge(pairs_df, df[['Issue key', 'Summary', 'Custom field (Story Points)']], left_on='Node 1', right_on='Issue key', how='left')
    merged_df = pd.merge(merged_df, df[['Issue key', 'Summary', 'Custom field (Story Points)']], left_on='Node 2', right_on='Issue key', how='left', suffixes=('_Node1', '_Node2'))

    return merged_df
