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

import streamlit_pandas_profiling
from streamlit_pandas_profiling import st_profile_report




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

st.title("Scrum Team Pulse")
st.subheader("Make data talk")


st.sidebar.title("Follow tabs")

tabs = ["Data", "Column Selector", "Chart Creation", "Dashboard", "Template Individual Performance", "Template Team Performance"]
current_tab = st.sidebar.radio("Select tab", tabs)


tab_descriptions = {
    "Data": "In the Data tab, you have the option to upload your project data, establish a connection to your Jira instance for real-time data retrieval, or utilize a sample dataset for exploratory purposes. Additionally, Pandas Profiling is available to facilitate comprehensive data understanding.",
    "Column Selector": "The Column Selector tab allows you to rename and select columns for your analysis. Furthermore, you can update the data frame based on your selected columns and seamlessly continue your analysis with the new dataframe",
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
    





if current_tab == "Data":
    # Create a radio button to choose the data source
    data_source = st.radio("Choose Data Source", ["Upload CSV", "Use Sample Data", "Connect to Jira Instance"])

    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader("Upload CSV File", type=['csv'])

        if uploaded_file:
            df = pd.read_csv(uploaded_file, encoding='iso-8859-1')
            st.session_state['data_frame'] = df
            st.write(df)
            st.success("Data successfully uploaded!")

            with st.expander("Profile data"):
                pr = df.profile_report()
                st.components.v1.html(pr.to_html(), width=900, height=600, scrolling=True)





    elif data_source == "Use Sample Data":
        # Load sample data from a repository or any other source
        # Replace the following line with code to load sample data
        df = pd.read_csv(r'sample_data/sample_data.csv')
        st.session_state['data_frame'] = df
        st.write(df)
        st.success("Sample data loaded successfully!")





        # Create an expander for the similarity functionality
        with st.expander("Similarity Functionality"):
            # Compute similarity and display results
            st.subheader("Similarity Results")

            # Add a slider for the similarity threshold
            threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.2, 0.05)  # Add a step parameter of 0.05

            # Wrap this code inside a Streamlit reactive block
            if st.checkbox("Update Similarity Based on Threshold"):
                df = preprocess_data(df)

                # Call the calculate_similarity function
                similar_pairs = calculate_similarity(df, threshold)

                # Display the threshold and the table with the pairs for the current threshold
                st.subheader(f"Similarity Threshold: {threshold}")
                st.dataframe(similar_pairs)


    elif data_source == "Connect to Jira Instance":
        jira_url = st.text_input("Jira URL", "https://ourJiraInstance.atlassian.net")
        jira_email = st.text_input("Jira Email", "")
        jira_token = st.text_input("Jira API Token", "", type="password")
        jql_query = st.text_area("Enter JQL Query", " ")

        if st.button("Fetch Data"):
            try:
                # Connect to Jira using the provided credentials
                jira = JIRA(server=jira_url, basic_auth=(jira_email, jira_token))
                st.success("Connected to Jira successfully!")

                # Fetch data using JQL query
                issues = jira.search_issues(jql_query, maxResults=None)  # Adjust maxResults as needed

                                # Add a button to compute similarity
                if st.button("Compute Similarity"):
                    # Compute similarity and display results
                    # You can use the code provided in the previous response to compute similarity
                    st.subheader("Similarity Results")

            except Exception as e:
                st.warning(f"Error fetching data from Jira: {e}")

                # Convert Jira issues to a DataFrame
                data = []
                for issue in issues:
                    # Customize the data extraction logic as per your requirements
                    # Example: Extract issue key, summary, and assignee
                    issue_data = {
                        'Issue Key': issue.key,
                        'Summary': issue.fields.summary,
                        'Assignee': issue.fields.assignee.name  # Fix the typo here (change '.ame' to '.name')
                        # Add more fields as needed
                    }
                    data.append(issue_data)

                df = pd.DataFrame(data)
                st.session_state['data_frame'] = df
                st.write(df)
                st.success("Data successfully fetched from Jira!")
            except Exception as e:
                st.warning(f"Error fetching data from Jira: {e}")


         
elif current_tab == "Column Selector":
    if 'data_frame' not in st.session_state:
        st.info("Please upload data first in the 'Data Upload' tab.")
    else:
        data_frame = st.session_state['data_frame']

        st.subheader("Data Table")
        st.dataframe(data_frame)

        st.subheader("Change Column Name")
        column_to_rename = st.selectbox("Select Column", data_frame.columns)
        new_column_name = st.text_input("New Column Name", "")
        if new_column_name:
            column_mapping = session_state.get_column_mapping()
            column_mapping[column_to_rename] = new_column_name
            session_state.set_column_mapping(column_mapping)

        st.subheader("Select Columns")
        selected_columns = st.multiselect("Select Columns", data_frame.columns)
        if selected_columns:
            new_data_frame = data_frame[selected_columns]
            st.dataframe(new_data_frame)
            if st.button("Update Data Frame"):
                st.session_state['data_frame'] = new_data_frame




elif current_tab == "Chart Creation":
    if 'data_frame' not in st.session_state:
        st.info("Please upload data first in the 'Data Upload' tab.")
    else:
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

                # Send a POST request to your Node-RED flow
                response = requests.post("http://127.0.0.1:1880/chatgpt", json=payload)

                # Check if the response was successful and display ChatGPT's response
                if response.status_code == 200:
                    chatgpt_response = response.json()
                    answer = chatgpt_response["answer"]
                    st.write("ChatGPT's Answer:", answer)
                else:
                    st.write("Error communicating with ChatGPT")
            # Inside the "Chart Creation" tab
            if st.button("Plot Chart"):
                # Send a request to Node-RED to trigger the data manipulation and chart generation
                response = requests.post("http://127.0.0.1:1880/trigger-data-manipulation")
                if response.status_code == 200:
                    st.success("Chart plotted successfully!")
                else:
                    st.error("Error plotting the chart. Please try again.")

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
        st.info("Please upload data first in the 'Data Upload' tab.")
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
            st.warning("Some required columns are missing in the uploaded data.")
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
        st.warning("Please upload and process the data first before accessing the examples.")





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
