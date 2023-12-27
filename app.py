import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder  


DATASET_PATH = 'C:\\Users\\sirto\\OneDrive\\Desktop\\Data Science\\smoking for report.csv'


@st.cache_data
def load_data():
    data = pd.read_csv(DATASET_PATH)
    return data


def preprocess_data(data):
    
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = data.select_dtypes(include=['object']).columns

    
    imputer_numeric = SimpleImputer(strategy='mean')
    data[numeric_cols] = imputer_numeric.fit_transform(data[numeric_cols])

    
    imputer_categorical = SimpleImputer(strategy='most_frequent')
    data[categorical_cols] = imputer_categorical.fit_transform(data[categorical_cols])

    return data


def plot_feature_distribution(data, feature):
    fig, ax = plt.subplots()
    sns.histplot(data[feature], kde=True, ax=ax)
    return fig


def perform_pca(data, n_components=2):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(data_scaled)
    return principal_components


def perform_kmeans(data, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(data)
    return kmeans


def train_classifier(X_train, y_train):
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    return clf


def evaluate_classifier(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    return report, cm


def main():
    st.title("Smoking Prevalence Analysis App")

   
    data = load_data()
    data = preprocess_data(data)

    
    label_encoder = LabelEncoder()
    categorical_cols = data.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        data[col] = label_encoder.fit_transform(data[col])

   
    with st.sidebar:
        st.header("Settings")
        feature_to_plot = st.selectbox("Select Feature to Plot", data.columns)
        n_clusters = st.slider("Select Number of Clusters for KMeans", 2, 10, 3)
        test_size = st.slider("Test Size for Model Training", 0.1, 0.5, 0.2)
    
    
    st.subheader("Data Visualization")
    if st.button('Show Feature Distribution'):
        fig = plot_feature_distribution(data, feature_to_plot)
        st.pyplot(fig)

    if st.button('Perform PCA and KMeans Clustering'):
        numeric_data = data.select_dtypes(include=['float64', 'int64'])
        principal_components = perform_pca(numeric_data)
        kmeans = perform_kmeans(principal_components, n_clusters)
        fig, ax = plt.subplots()
        ax.scatter(principal_components[:, 0], principal_components[:, 1], c=kmeans.labels_)
        st.pyplot(fig)

    st.subheader("Predictive Modelling")
    if st.checkbox("Train Random Forest Classifier"):
       
        X = data.drop('smoke', axis=1)
        y = data['smoke']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        clf = train_classifier(X_train, y_train)
        report, cm = evaluate_classifier(clf, X_test, y_test)
        st.text("Classification Report:")
        st.text(report)
        st.text("Confusion Matrix:")
        st.dataframe(cm)

if __name__ == "__main__":
    main()
