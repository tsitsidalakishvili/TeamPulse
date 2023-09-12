import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from jira import JIRA  # Import JIRA library

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
st.sidebar.title("Follow tabs")
tabs = ["Data Extraction", "Data Upload", "Column Selector", "Chart Creation", "Dashboard", "Examples"]
current_tab = st.sidebar.radio("Select tab", tabs)

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

if current_tab == "Data Upload":
    uploaded_file = st.file_uploader("Upload CSV File", type=['csv'])

    if uploaded_file:
        df = read_csv_files(uploaded_file)
        if df is not None:
            st.session_state['data_frame'] = df
            st.write(df)
            st.success("Data successfully uploaded!")
        else:
            st.warning("Failed to process the uploaded data.")

elif current_tab == "Data Extraction":
    st.sidebar.subheader("Connect to Jira Instance")
    jira_url = st.sidebar.text_input("Jira URL", "https://ourJiraInstance.atlassian.net")
    jira_email = st.sidebar.text_input("Jira Email", "")
    jira_token = st.sidebar.text_input("Jira API Token", "", type="password")
    jql_query = st.sidebar.text_area("Enter JQL Query", " ")

    if st.button("Fetch Data"):
        try:
            # Connect to Jira using the provided credentials
            jira = JIRA(server=jira_url, basic_auth=(jira_email, jira_token))
            st.success("Connected to Jira successfully!")

            # Fetch data using JQL query
            issues = jira.search_issues(jql_query, maxResults=None)  # Adjust maxResults as needed

            # Convert Jira issues to a DataFrame
            data = []
            for issue in issues:
                # Customize the data extraction logic as per your requirements
                # Example: Extract issue key, summary, and assignee
                issue_data = {
                    'Issue Key': issue.key,
                    'Summary': issue.fields.summary,
                    'Assignee': issue.fields.assignee.ame
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





elif current_tab == "Examples":
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






