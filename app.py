import streamlit as st
import pandas as pd
import pandas_profiling  # Import pandas-profiling
import plotly.express as px
import numpy as np
from jira import JIRA
import nltk
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from similarity import preprocess_data, calculate_similarity
import requests
import subprocess
import pandas as pd
from neo4j import GraphDatabase, basic_auth


import streamlit_pandas_profiling
from streamlit_pandas_profiling import st_profile_report

import neo4j
# Import Neo4jManager class
from neo4j_integration import Neo4jManager

import io  # Import the 'io' module for working with byte streams

import csv
# Function to create nodes from CSV



def create_nodes_from_csv(neo4j_manager, uploaded_file, label):
    # Check if a file was uploaded
    if uploaded_file is not None:
        # Read the uploaded CSV file as bytes
        file_contents = uploaded_file.read()

        # Convert the bytes to a string (assuming UTF-8 encoding)
        csv_content = file_contents.decode('utf-8')

        # Use 'io.StringIO' to create a file-like object from the string
        csv_file = io.StringIO(csv_content)

        # Now, you can read the CSV data from the 'csv_file' file-like object
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            try:
                # Attempt to convert 'Issue key' to an integer
                issue_key = int(row['Issue key'])
            except ValueError:
                # Handle the case where 'Issue key' is not a valid integer
                st.warning(f"Skipping row with invalid 'Issue key': {row}")
                continue

            # Assuming your CSV has columns like 'name', 'age', 'city'
            properties = {
                'Assignee': row['Assignee'],
                'Issue key': int(row['age']),
                'city': row['city']
            }
            neo4j_manager.create_node(label, properties)
    else:
        st.warning("Please upload a CSV file before creating nodes.")

            


def create_neo4j_driver():
    return GraphDatabase.driver(
        "bolt://3.86.204.9:7687",  # Update with your Neo4j hostname and port
        auth=basic_auth("neo4j", "preference-transaction-revision")  # Replace with your Neo4j username and password
    )


def execute_cypher_query(driver, cypher_query, params=None):
    with driver.session() as session:
        result = session.run(cypher_query, params)
        return result.data()







# Download stopwords if not already downloaded
nltk.download('stopwords')

# Set Streamlit configurations
st.set_page_config(layout="wide")  # Use the wide layout

def run():
    if 'session_state' not in st.session_state:
        st.session_state['session_state'] = _SessionState()

    session_state = st.session_state['session_state']

# Define session state
class _SessionState:
    def __init__(self):
        """Initialize session state."""
        self._charts = []
        self._column_mapping = {}  # To store column name changes

    def add_chart(self, chart):
        """Add a chart to the session state."""
        self._charts.append(chart)

    def get_charts(self):
        """Get all charts in the session state."""
        return self._charts

    def get_column_mapping(self):
        """Get the column name mapping."""
        return self._column_mapping

    def set_column_mapping(self, column_mapping):
        """Set the column name mapping."""
        self._column_mapping = column_mapping

if 'session_state' not in st.session_state:
    st.session_state['session_state'] = _SessionState()

session_state = st.session_state['session_state']

st.title("Make Data Talk")




st.sidebar.title("Follow tabs")

tabs = ["Data Source", "Column Selector", "Chart Creation", "Dashboard", "Template Individual Performance", "Template Team Performance"]
current_tab = st.sidebar.radio("Select tab", tabs)


tab_descriptions = {
    "Data": "In the Data tab, you can upload your project data or connect to your Jira instance to fetch real-time data.",
    "Column Selector": "The Column Selector tab allows you to rename and select columns for your analysis.",
    "Chart Creation": "In the Chart Creation tab, you can create charts to visualize your project's progress using AI assistance or by creating charts yourself.",
    "Dashboard": "The Dashboard tab displays all the charts you've created, providing an overview of your project.",
    "Template Individual Performance": "This tab offers pre-designed charts and insights to assess individual performance.",
    "Template Team Performance": "Here, you can access pre-designed charts to assess your team's performance efficiently."
}

# Create an expander for the tab description
with st.sidebar.expander("Tab Description"):
    if current_tab in tab_descriptions:
        st.write(tab_descriptions[current_tab])





def assignee_median_capacity(df, assignee_name):
    # Assuming a median story points computation is based on the 'Story Points' column
    median_points = df[df['Assignee'] == assignee_name]['Custom field (Story Points)'].median()
    return median_points

def read_csv_files(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file, encoding='iso-8859-1')
        return df
    except Exception as e:
        st.warning(f"Error reading the file: {e}")
        return None
    



def profile_data_func(df):
    with st.expander("Profile data"):
        pr = df.profile_report()
        st.components.v1.html(pr.to_html(), width=900, height=600, scrolling=True)

def similarity_func(df):
    with st.expander("Similarity Functionality"):
        st.subheader("Similarity Results")
        threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.2, 0.05)
        if st.checkbox("Update Similarity Based on Threshold"):
            df = preprocess_data(df)
            similar_pairs = calculate_similarity(df, threshold)
            st.subheader(f"Similarity Threshold: {threshold}")
            st.dataframe(similar_pairs)

if current_tab == "Data Source":
    data_source = st.radio("Choose Data Source", ["Upload CSV", "Use Sample Data", "Connect to Jira Instance", "Neo4j"])

    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader("Upload CSV File", type=['csv'])
        if uploaded_file:
            df = pd.read_csv(uploaded_file, encoding='iso-8859-1')
            st.session_state['data_frame'] = df
            st.write(df)
            st.success("Data successfully uploaded!")
            profile_data_func(df)
            similarity_func(df)

    elif data_source == "Use Sample Data":
        df = pd.read_csv(r'sample_data/sample_data.csv')
        st.session_state['data_frame'] = df
        st.write(df)
        st.success("Sample data loaded successfully!")
        profile_data_func(df)
        similarity_func(df)
        
    elif data_source == "Neo4j":
        st.title("Create Neo4j Model")

        # Get Neo4j connection details from the user
        neo4j_uri = st.text_input("Neo4j URI")
        neo4j_username = st.text_input("Neo4j Username")
        neo4j_password = st.text_input("Neo4j Password", type="password")

        # Allow the user to upload a CSV file
        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

        if uploaded_file:
            if neo4j_uri and neo4j_username and neo4j_password:
                # Create a Neo4jManager instance with user-provided credentials
                neo4j_manager = Neo4jManager(neo4j_uri, neo4j_username, neo4j_password)

                # Create nodes from the uploaded CSV file
                label = "Person"  # Set the label for the nodes
                create_nodes_from_csv(neo4j_manager, uploaded_file, label)

                # Close the Neo4j connection
                neo4j_manager.close()

                st.success("Nodes created successfully!")
            else:
                st.warning("Please provide Neo4j connection details (URI, username, and password) to create nodes.")




        st.subheader("Neo4j Cypher Query")

        cypher_query = st.text_area("Enter Cypher Query")

        if st.button("Execute Cypher Query"):
            if cypher_query:
                #Execute the query
                with create_neo4j_driver() as driver:
                    results = execute_cypher_query(driver, cypher_query)
                #Display the results
                if results:
                    st.subheader("Query Results")
                    st.write(results)
                else:
                    st.warning("No results returned.")
            else:
                st.warning("Please enter a Cypher query.")


        
            



    elif data_source == "Connect to Jira Instance":
        with st.expander("Use JIra REST API"):
            jira_url = st.text_input("Jira URL", "https://tsitsieigen.atlassian.net")
            jira_email = st.text_input("Your email here", "")
            jira_token = st.text_input("Jira API Token", "", type="password")
            jql_query = st.text_area("", " ")

            if st.button("Fetch Data"):
                try:
                    jira = JIRA(server=jira_url, basic_auth=(jira_email, jira_token))
                    st.success("Connected to Jira successfully!")
                    issues = jira.search_issues(jql_query, maxResults=None)
                    issues_data = []
                    for issue in issues:
                        issue_dict = {
                            "Key": issue.key,
                            "Summary": issue.fields.summary,
                            "Status": issue.fields.status.name,
                            "Assignee": issue.fields.assignee.displayName if issue.fields.assignee else "Unassigned",
                        }
                        issues_data.append(issue_dict)

                    df = pd.DataFrame(issues_data)
                    st.session_state['data_frame'] = df
                    st.table(df)

                except Exception as e:
                    st.error(f"An error occurred: {e}")
                #profile_data_func(df)
                #similarity_func(df)


        with st.expander("Use Node-Red flow"):

            if st.button('Start Node-RED flow'):
                response = requests.post('http://127.0.0.1:1880/triggerFlow')
                if response.status_code == 200:
                    st.write('Flow started successfully!')
                    # Fetching the data immediately after triggering the flow
                    data_response = requests.get('http://127.0.0.1:1880/fetchTransformedData')
                    if data_response.status_code == 200:
                        data = data_response.json()
                        if isinstance(data, list) and all(isinstance(item, dict) for item in data):
                            df = pd.DataFrame(data)
                            st.write(df)
                        else:
                            st.warning('Received unexpected data format from Node-RED.')
                    else:
                        st.error('Error fetching the initial data.')
                else:
                    st.error('Error starting the flow.')








elif current_tab == "Column Selector":
    # Fetch data from Node-RED if it's not in session state
    if 'data_frame' not in st.session_state:
        try:
            response = requests.get('http://127.0.0.1:1880/fetchTransformedData')
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list) and all(isinstance(item, dict) for item in data):
                    st.session_state['data_frame'] = pd.DataFrame(data)
                else:
                    st.warning('Received unexpected data format from Node-RED.')
            else:
                st.error('Error fetching data from Node-RED.')
        except Exception as e:
            st.error(f"An error occurred while fetching data from Node-RED: {e}")

    # Existing Data Frame Availability Check
    if 'data_frame' not in st.session_state:
        st.info("Please upload data in the 'Data Upload' tab, use the sample data, connect to the Jira Instance, or fetch data from Node-RED.")
    else:
        data_frame = st.session_state['data_frame']
        st.subheader("Data Table")
        st.dataframe(data_frame)

        st.subheader("Change Column Name")
        column_to_rename = st.selectbox("Select Column", data_frame.columns)
        new_column_name = st.text_input("New Column Name", "")
        if new_column_name:
            column_mapping = session_state.get_column_mapping()  # Assuming you've a function or method for this
            column_mapping[column_to_rename] = new_column_name
            session_state.set_column_mapping(column_mapping)  # Assuming you've a function or method for this

        st.subheader("Select Columns")
        selected_columns = st.multiselect("Select Columns", data_frame.columns)
        if selected_columns:
            new_data_frame = data_frame[selected_columns]
            st.dataframe(new_data_frame)
            if st.button("Update Data Frame"):
                st.session_state['data_frame'] = new_data_frame


elif current_tab == "Chart Creation":
    if 'data_frame' not in st.session_state:
        st.info("Please upload data first in the 'Data Upload' tab or use sample data.")
    else:
        # Display the data table
        data_frame = st.session_state['data_frame']
        st.subheader("Data Table")
        st.dataframe(data_frame)
        
        # ChatGPT Integration Extender
        with st.expander("Use AI Assistance (LLM)"):
            # Create an input text box for user questions
            user_question = st.text_input("Enter your question:")

            if st.button("Submit"):
                # Define the payload to send to Node-RED
                payload = {
                    "question": user_question,
                    "topic": ""
                }
        try:
            response = requests.post("http://127.0.0.1:1880/chatgpt", json=payload)
            # Check the response status code and handle any errors here
            if response.status_code == 200:
                chatgpt_response = response.json()
                answer = chatgpt_response["answer"]
                st.write("ChatGPT's Answer:", answer)
            else:
                st.write("Error communicating with ChatGPT")
        except Exception as e:
            st.write("An error occurred during the HTTP request:", str(e))



        # Create chart yourself Extender
        with st.expander("Create chart yourself"):
            data_frame = st.session_state['data_frame']

            column_mapping = session_state.get_column_mapping()
            for old_column, new_column in column_mapping.items():
                if old_column in data_frame.columns:
                    data_frame = data_frame.rename(columns={old_column: new_column})

            column_x = st.selectbox("Select X-axis variable", data_frame.columns)
            column_y = st.selectbox("Select Y-axis variable", data_frame.columns)
            chart_type = st.selectbox("Select chart type", ['Bar', 'Line', 'Scatter', 'Histogram', 'Pie'])

            # Color Grouping
            # Using 'Choose legend' and adding a tooltip for clarity
            column_color = st.selectbox("Choose legend", ['None'] + list(data_frame.columns), help="Select a column to differentiate data by color on the chart.")
            
            # Hover Data
            hover_data = st.multiselect("Select additional columns for hover data", data_frame.columns, default=[])

            # Facet Charts
            facet_col = st.selectbox("Choose column for faceting (creates multiple charts)", ['None'] + list(data_frame.columns))

            # Scatter size
            scatter_size = None
            if chart_type == 'Scatter':
                scatter_size = st.selectbox("Choose column for scatter point size (Optional)", ['None'] + list(data_frame.columns))

            # Summation of a specific column
            if st.checkbox("Display sum of a column"):
                column_to_sum = st.selectbox("Select column to sum", data_frame.select_dtypes(include=[np.number]).columns)
                if column_to_sum:
                    summed_value = data_frame[column_to_sum].sum()
                    st.markdown(f"**Total sum of {column_to_sum}:** {summed_value}")
                    
                    # Optional: Visualize the sum
                    if st.checkbox("Visualize the sum"):
                        fig_sum = px.bar(x=[column_to_sum], y=[summed_value], title=f"Total sum of {column_to_sum}")
                        st.plotly_chart(fig_sum)

                title = st.text_input("Chart Title", "Your Chart Title")
                x_axis_label = st.text_input("X-axis Label", column_x)
                y_axis_label = st.text_input("Y-axis Label", column_y)

                if chart_type == 'Bar':
                    fig = px.bar(data_frame, x=column_x, y=column_y, color=column_color if column_color != 'None' else None, 
                                hover_data=hover_data, facet_col=facet_col if facet_col != 'None' else None,
                                title=title, labels={'x': x_axis_label, 'y': y_axis_label})

                elif chart_type == 'Line':
                    fig = px.line(data_frame, x=column_x, y=column_y, color=column_color if column_color != 'None' else None, 
                                hover_data=hover_data, facet_col=facet_col if facet_col != 'None' else None,
                                title=title, labels={'x': x_axis_label, 'y': y_axis_label})

                elif chart_type == 'Scatter':
                    scatter_size_data = None
                    if scatter_size and scatter_size != 'None':
                        if pd.api.types.is_numeric_dtype(data_frame[scatter_size]):
                            scatter_size_data = data_frame[scatter_size]
                        else:
                            st.warning(f"Column {scatter_size} is not numeric. Size won't be applied to scatter plot.")
                            
                    fig = px.scatter(data_frame, x=column_x, y=column_y, 
                                    color=column_color if column_color != 'None' else None,
                                    hover_data=hover_data, 
                                    facet_col=facet_col if facet_col != 'None' else None,
                                    size=scatter_size_data,  # Use the validated scatter size data here
                                    title=title, labels={'x': x_axis_label, 'y': y_axis_label})

                elif chart_type == 'Histogram':
                    fig = px.histogram(data_frame, x=column_x, color=column_color if column_color != 'None' else None,
                                    hover_data=hover_data, title=title, labels={'x': x_axis_label, 'y': y_axis_label})

                elif chart_type == 'Pie':
                    fig = px.pie(data_frame, names=column_x, values=column_y, title=title)

                st.plotly_chart(fig)

                if st.button("Add to dashboard"):
                    session_state.add_chart(fig)
                    st.success("Chart added to the dashboard!")


elif current_tab == "Dashboard":
    dashboard_charts = session_state.get_charts()

    if not dashboard_charts:
        st.info("No charts added to the dashboard yet.")
    else:
        st.subheader('Dashboard')
        for chart in dashboard_charts:
            st.plotly_chart(chart)



elif current_tab == "Template Team Performance":
    if 'data_frame' not in st.session_state:
        st.info("Please upload data first in the 'Data Upload' tab or load sample data before accessing the examples.")
    else:
        data_frame = st.session_state['data_frame']

        st.subheader("Team Average Ratio by Sprint")

        # Line Chart - Team Average Ratio by Sprint
        team_avg_ratio_by_sprint = data_frame.groupby('Sprint')['Avg_Ratio'].mean().reset_index()
        team_avg_ratio_by_sprint_chart = px.line(
            team_avg_ratio_by_sprint,
            x='Sprint',
            y='Avg_Ratio',
            title='Team Average Ratio by Sprint',
            labels={'Avg_Ratio': 'Average Ratio', 'Sprint': 'Sprint'},
        )

        st.plotly_chart(team_avg_ratio_by_sprint_chart, use_container_width=True)

        st.subheader("Team Capacity in Sprint")

        # Horizontal Bar Chart - Team Capacity in Sprint
        team_capacity_by_sprint = data_frame.groupby('Sprint')['Assignee Capacity'].sum().reset_index()
        team_capacity_by_sprint_chart = px.bar(
            team_capacity_by_sprint,
            x='Assignee Capacity',
            y='Sprint',
            orientation='h',
            title='Team Capacity in Sprint',
            labels={'Assignee Capacity': 'Capacity', 'Sprint': 'Sprint'},
        )

        st.plotly_chart(team_capacity_by_sprint_chart, use_container_width=True)


elif current_tab == "Template Individual Performance":
    if 'data_frame' in st.session_state:
        df = st.session_state['data_frame']

        # Check for required columns
        required_columns = ['Custom field (Story Points)', 'Sprint', 'Project name', 'Assignee', 'Time Spent', 'Issue Type', 'Component']
        if not all(col in df.columns for col in required_columns):
            st.info("Please upload data first in the 'Data Upload' tab or load sample data.")

        else:
            # Derived columns
            df['days'] = df['Time Spent'] / 28800
            df['Avg_Ratio'] = df['Custom field (Story Points)'] / df['days']

            # Radar Chart - Average Ratio by Assignee and Component
            avg_ratio_assignee_component = df.groupby(['Assignee', 'Component'])['Avg_Ratio'].mean().reset_index()
            radar_chart_assignee_component = px.line_polar(avg_ratio_assignee_component, r='Avg_Ratio', theta='Component',
                                                    line_close=True,
                                                    title='Average Ratio by Assignee and Component',
                                                    labels={'Avg_Ratio': 'Average Ratio', 'Component': 'Component'},
                                                    color='Assignee')

            # Boxplot - Assignee Capacity in Sprint
            df['Assignee Capacity'] = df['Assignee'].apply(lambda x: assignee_median_capacity(df, x))
            assignee_capacity_fig = px.box(df, x='Assignee', y='Assignee Capacity', title='Assignee Capacity in Sprint')
            assignee_capacity_fig.update_traces(boxpoints='all', jitter=0.3, pointpos=-1.8, hovertemplate='Capacity: %{y}<br><extra></extra>')
            assignee_capacity_fig.update_yaxes(range=[0, df['Assignee Capacity'].max() + 10])

            # Bar Chart - Average Ratio by Issue Type and Assignee
            avg_ratio_data = df.groupby(['Issue Type', 'Assignee'])['Avg_Ratio'].mean().reset_index()
            avg_ratio_data.loc[avg_ratio_data['Issue Type'].isin(['Task', 'Sub-task']), 'Issue Type'] = 'Task & Sub-task'
            filtered_avg_ratio_data = avg_ratio_data[avg_ratio_data['Issue Type'].isin(['Task & Sub-task', 'Bug'])]
            color_map = {'Bug': 'darkred', 'Task & Sub-task': 'blue'}
            avg_ratio_chart = px.bar(
                filtered_avg_ratio_data,
                x='Assignee',
                y='Avg_Ratio',
                color='Issue Type',
                barmode='group',
                color_discrete_map=color_map,
                title='Average Ratio by Issue Type and Assignee'
            )
            avg_ratio_chart.update_traces(texttemplate='%{value:.2f}', textposition='inside')

            # Line Chart - Average Ratio by Sprint and Assignee
            line_chart_avg_ratio = px.line(
                df.groupby(['Sprint', 'Assignee'])['Avg_Ratio'].mean().reset_index(),
                x='Sprint',
                y='Avg_Ratio',
                color='Assignee',
                title='Average Ratio by Sprint and Assignee'
            )

        # Adjust Plotly figure margins to 1
        radar_chart_assignee_component.update_layout(margin=dict(t=50, b=30, l=30, r=30))
        radar_chart_assignee_component.update_layout(margin=dict(t=45, b=45, l=45, r=45))
        assignee_capacity_fig.update_layout(margin=dict(t=45, b=45, l=45, r=45))
        avg_ratio_chart.update_layout(margin=dict(t=45, b=45, l=45, r=45))
        line_chart_avg_ratio.update_layout(margin=dict(t=45, b=45, l=45, r=45))




        # Setting up the 2x2 grid layout using Streamlit's columns
        col1, col2 = st.columns(2)

        # 1st Row: First two charts
        with col1:
            st.plotly_chart(radar_chart_assignee_component, use_container_width=True)
        with col2:
            st.plotly_chart(assignee_capacity_fig, use_container_width=True)

        # Use st.empty() for spacing adjustment (if needed)
        st.empty()

        # 2nd Row: Last two charts
        with col1:
            st.plotly_chart(avg_ratio_chart, use_container_width=True)
        with col2:
            st.plotly_chart(line_chart_avg_ratio, use_container_width=True)

    else:
        st.info("Please upload data first in the 'Data Upload' tab or load sample data before accessing the examples.")







if __name__ == "__main__":
    run()


# Create a spacer to push content to the bottom of the sidebar
st.sidebar.markdown("---")

# Add the "Created by" line at the bottom of the sidebar
st.sidebar.markdown("Created by [Tsitsi Dalakishvili](https://www.linkedin.com/in/tsitsi-dalakishvili/)")



if 'command' in st.session_state:
    command = st.session_state['command']
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    st.session_state['command_result'] = {
        'returncode': result.returncode,
        'stdout': result.stdout.decode('utf-8'),
        'stderr': result.stderr.decode('utf-8'),
    }

if 'command_result' in st.session_state:
    result = st.session_state['command_result']
    st.subheader("Command Execution Result")
    st.write(f"Return Code: {result['returncode']}")
    st.write(f"Standard Output:\n{result['stdout']}")
    st.write(f"Standard Error:\n{result['stderr']}")
