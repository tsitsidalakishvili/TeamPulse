import streamlit as st
import pandas as pd
import datetime
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from io import BytesIO
import re
import datetime
import numpy as np
import datetime
import pandas as pd
import logging
from streamlit.components import v1 as components



def read_csv_files(uploaded_file_iterative, uploaded_file_eigen):
    try:
        Iterative = pd.read_csv(uploaded_file_iterative, encoding='iso-8859-1')
        Eigen = pd.read_csv(uploaded_file_eigen, encoding='iso-8859-1')
        return Iterative, Eigen
    except Exception as e:
        st.warning(f"Error reading the files: {e}")
        return None, None

sprint_bins = [
    ('09 Nov 2022', '30 Nov 2022'),
    ('30 Nov 2022', '21 Dec 2022'),
    ('04 Jan 2023', '25 Jan 2023'),
    ('25 Jan 2023', '15 Feb 2023'),
    ('15 Feb 2023', '08 Mar 2023'),
    ('08 Mar 2023', '29 Mar 2023'),
    ('29 Mar 2023', '19 Apr 2023'),
    ('19 Apr 2023', '10 May 2023'),
    ('10 May 2023', '31 May 2023'),
    ('31 May 2023', '21 June 2023'),
    ('21 June 2023', '12 July 2023'),
    ('12 July 2023', '2 August 2023'),
    ('2 August 2023', '23 August 2023'),
    ('23 August 2023', '13 September 2023')

]

# Convert the date strings to datetime objects
sprint_bins = [(pd.to_datetime(start), pd.to_datetime(end)) for start, end in sprint_bins]


def get_sprint(date, sprint_bins):
    for i, (start_date, end_date) in enumerate(sprint_bins):
        if start_date <= date <= end_date:
            return f"Sprint {80 + i}"
    return None



def extract_key_ID(df):
    df = df.copy()  # Make a copy to work with
    
    # Extracting the pattern after F# inside []
    df['Issue key'] = df['Issue summary'].str.extract('\[F#(\d+)\]', expand=False)
    
    # For rows where 'Issue key' is NaN, extracting the pattern inside []
    mask = df['Issue key'].isna()
    df.loc[mask, 'Issue key'] = df.loc[mask, 'Issue summary'].str.extract('\[(.*?)\]', expand=False)
    
    # For rows where 'Issue key' is still NaN, taking the entire 'Issue summary' string
    mask = df['Issue key'].isna()
    df.loc[mask, 'Issue key'] = df.loc[mask, 'Issue summary']

    df = df.drop('Issue summary', axis=1)
    return df





def preprocess_data(Iterative, Eigen):
    # For Iterative dataframe
    if 'Issue summary' in Iterative.columns:
        Iterative = Iterative[['Issue summary', 'Hours', 'Full name', 'Work date', 'CoreTime', 'Issue Type', 'Issue Status']]
        Iterative = extract_key_ID(Iterative)
        Iterative = Iterative.rename(columns={'Full name': 'Assignee'})
    else:
        st.warning("Issue summary not found in Iterative file.")
        return None, None

    # For Eigen dataframe
    eigen_columns = ['Summary', 'Issue key', 'Status', 'Resolved', 'Epic Link Summary', 'Labels', 'Priority', 'Issue Type', 
                    'Assignee', 'Creator', 'Sprint', 'Created', 'Resolution', 'Custom field (Story Points)',
                    'Custom field (CoreTimeActivity)', 'Custom field (CoreTimeClient)', 'Custom field (CoreTimePhase)', 
                    'Custom field (CoreTimeProject)', 'Project name']

    if all(col in Eigen for col in eigen_columns):
        Eigen = Eigen[eigen_columns]
        Eigen = Eigen.rename(columns={
            'Custom field (CoreTimeActivity)': 'CoreTimeActivity', 
            'Custom field (CoreTimeClient)': 'CoreTimeClient',
            'Custom field (CoreTimePhase)': 'CoreTimePhase', 
            'Custom field (CoreTimeProject)': 'CoreTimeProject', 
            'Epic Link Summary': 'Epic', 
            'Custom field (Story Points)': 'Story Points'
        })
    else:
        st.warning("Some necessary columns are missing in Eigen file.")
        return None, None

    return Iterative, Eigen



def impute_story_points_with_median(df):
    # Calculate the median of 'Story Points' from available data
    median_sp = df['Story Points'].median()

    # Impute missing values with the median
    df['Story Points'].fillna(median_sp, inplace=True)

    return df






def merge_data(Iterative, Eigen, sprint_bins):

    merged_df = pd.merge(Iterative, Eigen, on='Issue key', how='outer')

    merged_df = merged_df.rename(columns={'Assignee_x': 'Assignee', 'Issue Type_x': 'Issue Type'})
    merged_df = merged_df.drop('Issue Type_y', axis=1)
    merged_df = merged_df[['Issue key', 'Work date', 'Assignee', 'Summary','Sprint', 'Status', 'Priority', 'Project name', 'Issue Type',
                           'Creator', 'Created', 'Resolution', 'Story Points', 'Hours', 'Resolved', 'Epic', 'CoreTimeActivity', 
                           'CoreTimeClient', 'CoreTimePhase', 'CoreTimeProject', 'Labels']]
    

    grouped_df = merged_df.groupby('Issue key').agg({'Hours': 'sum', 'Story Points': 'first', 'Assignee': 'first', 'Summary': 'first',
                                                     'Sprint': 'first', 'Resolution': 'first', 'Resolved': 'first', 
                                                     'Created': 'first', 'Priority': 'first', 'Creator': 'first', 
                                                     'CoreTimeActivity': 'first', 'CoreTimeClient': 'first', 
                                                     'CoreTimePhase': 'first', 'Epic': 'first', 'CoreTimeProject': 'first', 
                                                     'Project name': 'first', 'Status': 'first', 'Labels': 'first', 
                                                     'Work date': 'first', 'Issue Type': 'first'}).reset_index()
    
    grouped_df['days'] = grouped_df['Hours'] / 8
    grouped_df['Avg_Ratio'] = grouped_df['Story Points'] / grouped_df['days']
    grouped_df['Work date'] = pd.to_datetime(grouped_df['Work date'])
    grouped_df['Created'] = pd.to_datetime(grouped_df['Created'])
    grouped_df['Sprint'] = grouped_df['Created'].apply(lambda x: get_sprint(x, sprint_bins))
    grouped_df = impute_story_points_with_median(grouped_df) # Apply imputation here

    pattern = r'^[A-Z]{3,}-\d+$' 
    non_matching_rows = grouped_df[~grouped_df['Issue key'].str.match(pattern)]

    meetings = non_matching_rows.groupby('Sprint').agg({'days': 'sum', 'Story Points': 'sum'}).reset_index()

    return grouped_df, meetings, non_matching_rows


def load_data(uploaded_file_iterative, uploaded_file_eigen, sprint_bins):
    if uploaded_file_iterative is not None and uploaded_file_eigen is not None:
        Iterative, Eigen = read_csv_files(uploaded_file_iterative, uploaded_file_eigen)
        if Iterative is not None and Eigen is not None:
            Iterative, Eigen = preprocess_data(Iterative, Eigen)

    if Iterative is not None and Eigen is not None:
        df, meetings, non_matching = merge_data(Iterative, Eigen, sprint_bins)
        return df, meetings  # Update here
    else:
        return None, None




#-------------------------------------------------------------------------------------------------------------------------------------#
DEFAULT_RATES = {
    "Pawel G": 42.80,
    "Marcin Ko": 42.80,                                               
    "Marcin Kl": 26.75,
    "Lukasz": 31.03,
    "Alek D": 26.75,
    "Dawid N": 26.75
}
# Constants
DEFAULT_BG_COLOR = 'rgba(0, 0, 0, 0)'

def get_assignee_rates(df):
    """
    Get assignee rates from the DataFrame.
    """
    assignee_rates = {}
    # Check for assignees with missing rates or assignees not in the dataframe and set them to default
    for assignee, default_rate in DEFAULT_RATES.items():
        if assignee not in assignee_rates or assignee_rates[assignee] is None:
            assignee_rates[assignee] = default_rate

    return assignee_rates




def display_assignee_rates(assignee_rates):
    """
    Display the rates of assignees in the sidebar.
    """
    for assignee, rate in assignee_rates.items():
        st.sidebar.write(f"Rate for Assignee {assignee}: {rate}")

def update_chart_layout(chart, title, height=None, width=None):
    """
    Update chart layout and styling.
    """
    chart.update_layout(
        height=height,
        width=width,
        margin=dict(l=0, r=0, t=50, b=0),
        plot_bgcolor=DEFAULT_BG_COLOR,
        paper_bgcolor=DEFAULT_BG_COLOR,
        shapes=[
            dict(
                type='rect',
                xref='paper',
                yref='paper',
                x0=0,
                y0=0,
                x1=1,
                y1=1,
                line=dict(color='black', width=1),
                fillcolor=DEFAULT_BG_COLOR
            )
        ],
        title=title
    )


def create_combined_chart(df, sprint_summary, sprint_avg_ratio):
    # Create the combined chart
    combined_chart = go.Figure()
    combined_chart.add_trace(go.Bar(
        x=sprint_summary['Sprint'],
        y=sprint_summary['Story Points'],
        name='Story Points',
        text=sprint_summary['Story Points'],
        textposition='auto'
    ))
    combined_chart.add_trace(go.Bar(
        x=sprint_summary['Sprint'],
        y=sprint_summary['days'],
        name='days',
        text=sprint_summary['days'],
        textposition='auto'
    ))

    # Update the chart layout and styling
    combined_chart.update_layout(
        barmode='group',
        title='Delivered Story Points vs. Worked days by Sprint',
        xaxis=dict(title='Sprint'),
        yaxis=dict(title='Value'),  # Changed the y-axis title since it's no longer normalized
        legend=dict(title='Metrics'),
        font=dict(color='black')
    )

    # Apply styling to the bars and markers
    for trace in combined_chart.data:
        if 'marker' in trace and 'line' in trace['marker']:
            trace.marker.line.color = 'black'
        if 'textfont' in trace:
            trace.textfont.color = 'black'
            trace.textfont.size = 14

    return combined_chart

def display_tab1(df, assignee_rates):
    """
    Display visualizations and data for Tab 1.
    """
    df['Work date'] = pd.to_datetime(df['Work date'], infer_datetime_format=True)
    df['Cost'] = df['Hours'] * df['Assignee'].map(assignee_rates)

    epic_cost_data = df.groupby(['Epic', 'CoreTimeClient'])['Cost'].sum().reset_index()
    epic_sum_data = epic_cost_data.groupby('Epic')['Cost'].sum().reset_index()

    def select_data(trace, points, selector):
        """ Callback function to capture selected data points. """
        indices = points.point_inds
        if indices:
            index = indices[0]
            selected_epic = trace.customdata[index][0]
            st.session_state.selected_epic = selected_epic

    epic_hours_fig = px.bar(
        epic_cost_data,
        x='Cost',
        y='Epic',
        color='CoreTimeClient',
        orientation='h',
        labels={'Cost': '', 'Epic': 'Epic', 'CoreTimeClient': 'CoreTimeClient'},
        title='Cost by Epics'
    )
    epic_hours_fig.update_traces(customdata=epic_cost_data.values, selector=dict(type='bar'))

    for trace in epic_hours_fig.data:
        trace.on_click(select_data)

    max_cost = epic_sum_data['Cost'].max()
    for i, row in epic_sum_data.iterrows():
        x = max_cost * 1.01
        y = row['Epic']
        text = f"{row['Cost']:.2f} €"
        epic_hours_fig.add_annotation(
            x=x,
            y=y,
            text=text,
            showarrow=False,
            font=dict(color='white'),
            xshift=10,
            align='left'
        )
    epic_hours_fig.update_layout(
        height=600,
        width=1400,
        margin=dict(l=0, r=0, t=50, b=0),
        plot_bgcolor='rgba(0, 0, 0, 0)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        shapes=[
            dict(
                type='rect',
                xref='paper',
                yref='paper',
                x0=0,
                y0=0,
                x1=1,
                y1=1,
                line=dict(color='white', width=1),
                fillcolor='rgba(0, 0, 0, 0)'
            )
        ]
    )

    treemap_data = df.groupby(['Sprint', 'Assignee', 'CoreTimeClient', 'CoreTimeProject', 'CoreTimePhase', 'CoreTimeActivity'])['Cost'].sum().reset_index()
    treemap_fig = px.treemap(
        treemap_data,
        path=['Sprint', 'Assignee','CoreTimeClient', 'CoreTimeProject', 'CoreTimePhase', 'CoreTimeActivity'],
        values='Cost',
        title='Cost by CoreTimeClient, Project, Phase, and Activity'
    )
    treemap_fig.update_layout(
        height=600,
        width=800,
        margin=dict(l=0, r=0, t=50, b=0),
        plot_bgcolor='rgba(0, 0, 0, 0)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        shapes=[
            dict(
                type='rect',
                xref='paper',
                yref='paper',
                x0=0,
                y0=0,
                x1=1,
                y1=1,
                line=dict(color='black', width=1),
                fillcolor='rgba(0, 0, 0, 0)'
            )
        ]
    )
    
    line_chart_story_points = px.line(
        df.groupby(['Work date', 'CoreTimeClient'])['Story Points'].sum().reset_index(),
        x='Work date',
        y='Story Points',
        color='CoreTimeClient',
        labels={'Work date': 'Week', 'Story Points': 'Story Points', 'CoreTimeClient': 'Client'},
        title='Story Points Delivered Weekly by Client'
    )
    line_chart_cost = px.line(
        df.groupby(['Work date', 'CoreTimeClient'])['Cost'].sum().reset_index(),
        x='Work date',
        y='Cost',
        color='CoreTimeClient',
        labels={'Work date': 'Week', 'Cost': 'Cost', 'CoreTimeClient': 'Client'},
        title='Weekly Cost by Client'
    )


    # Outer container# Outer container
    outer_container = st.container()
    
    with outer_container:
        container2 = st.container()
        col2, col3 = container2.columns(2)

        col2.plotly_chart(epic_hours_fig, use_container_width=True)
        col2.plotly_chart(line_chart_cost, use_container_width=True)
        
        col3.plotly_chart(treemap_fig, use_container_width=True)
        col3.plotly_chart(line_chart_story_points, use_container_width=True)

    search_value = st.text_input("Search for value in table rows:", "")

    # Filter the DataFrame based on the search input
    filtered_df = df[df.apply(lambda row: search_value.lower() in str(row).lower(), axis=1)] if search_value else df
    # Display the filtered DataFrame as a table
    st.dataframe(filtered_df)
#-----------------------------------------------------------------------------------------------------------------------#

def calculate_average_ratio_by_project(df):
    avg_ratio_by_project = df.groupby('CoreTimeProject')['Avg_Ratio'].mean().reset_index()
    return avg_ratio_by_project

def create_cfd_chart(avg_ratio_by_project):
    # Sort the data by CoreTimeProject to ensure it appears in the desired order on the chart
    avg_ratio_by_project = avg_ratio_by_project.sort_values(by='CoreTimeProject')

    # Create the Cumulative Flow Diagram
    cfd_chart = go.Figure()
    cfd_chart.add_trace(go.Scatter(
        x=avg_ratio_by_project['CoreTimeProject'],
        y=avg_ratio_by_project['Avg_Ratio'],
        mode='lines+markers',
        line=dict(color='blue'),
        marker=dict(symbol='circle', size=8, color='blue'),
        name='Average Ratio by CoreTimeProject'
    ))

    # Update the chart layout and styling
    cfd_chart.update_layout(
        title='Team Average Ratio by CoreTimeProject',
        xaxis=dict(title='CoreTimeProject'),
        yaxis=dict(title='Average Ratio'),
        font=dict(color='black')
    )

    # Apply styling to the markers
    cfd_chart.data[0].marker.line.color = 'black'

    return cfd_chart


def generate_word_cloud_from_file(file_path):
        # Read the text data from the file
        with open(file_path, 'r') as file:
            text_data = file.read()

        # Create a WordCloud object
        wordcloud = WordCloud(width=800, height=400, background_color='black').generate(text_data)

        # Generate the word cloud image and return it
        plt.figure(figsize=(8, 4))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')  # Hide axes
        plt.tight_layout()

        # Save the word cloud to a BytesIO object
        word_cloud_data = BytesIO()
        plt.savefig(word_cloud_data, format='png')
        plt.close()

        # Rewind the BytesIO object
        word_cloud_data.seek(0)

        return word_cloud_data



def integrate_lyris_iframe():
    lyris_iframe = """
    <iframe src="https://lyris.ai/assets/public/html/iframeLyris.html?api-key=XvuccCm1mDHbenCE8mXQd69TYLAgsH0de10skhu2n1UAvQMG4LGqobR1IFACv7iTo1l2tgrlAQjBY1GbrPxLPRjZe7rQdE5YWkVl" 
    width="100%" height="100" frameborder="0"></iframe>
    """
    #components.html(lyris_iframe, height=100)




def display_tab2(df, assignee_rates, meetings=None):
    integrate_lyris_iframe()  # Call the function to integrate Lyris.

    # Remove whitespace from column names (if any)
    df.columns = df.columns.str.strip()

    # Filter the DataFrame to only include rows where 'Done' is True
    done_df = df[df['Status'] == 'Done']

    # Group by 'Sprint' and aggregate based on the 'Story Points' and 'days' columns
    sprint_summary = done_df.groupby('Sprint').agg({'Story Points': 'sum', 'days': 'sum'}).reset_index()


        # Calculate the total story points and total days for each sprint
    sprint_totals = df.groupby('Sprint').agg({'Story Points': 'sum', 'days': 'sum'}).reset_index()

    # Calculate the average ratio as story points divided by days for each sprint
    sprint_totals['Avg_Ratio'] = sprint_totals['Story Points'] / sprint_totals['days']

    # Then, use the sprint_totals DataFrame for plotting the Team Average Ratio by Sprint
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=sprint_totals['Sprint'],
        y=sprint_totals['Avg_Ratio'],
        mode='lines+markers',
        marker=dict(size=8),
        line=dict(width=2),
    ))



    # Update layout for the line chart
    fig2.update_layout(
        title='Team Average Ratio by Sprint',  # Add title for the line chart
        xaxis=dict(title='Sprint'),
        yaxis=dict(title='Average Ratio'),
        height=500,
        margin=dict(l=100, r=100, t=100, b=100),
    )


    avg_ratio_assignee_phase = df.groupby(['Assignee', 'CoreTimePhase'])['Avg_Ratio'].mean().reset_index()

    fig3 = go.Figure()

    # Add traces for each Assignee
    for assignee in avg_ratio_assignee_phase['Assignee'].unique():
        assignee_data = avg_ratio_assignee_phase[avg_ratio_assignee_phase['Assignee'] == assignee]
        fig3.add_trace(go.Scatterpolar(
            r=assignee_data['Avg_Ratio'],
            theta=assignee_data['CoreTimePhase'],
            mode='lines+markers',
            name=assignee,
            hovertemplate='%{theta}: %{r:.2f}<extra></extra>',
        ))

    # Update layout for the Radar Chart
    fig3.update_layout(
        title='Average Ratio by Assignee and Core Time Phase',  # Add title for the radar chart
        polar=dict(
            radialaxis=dict(
                title='Average Ratio',
                tickmode='linear',
                tickvals=[0, 0.5, 1],
            ),
            angularaxis=dict(
                tickmode='array',
                tickvals=avg_ratio_assignee_phase['CoreTimePhase'].unique(),
            ),
        ),
        showlegend=True,
        legend=dict(
            title='Assignee',
            traceorder='normal',
            font=dict(size=10),
        ),
        height=500,
        margin=dict(l=100, r=100, t=100, b=100),
    )

    # Calculate average ratio by CoreTimeProject
    avg_ratio_by_project = calculate_average_ratio_by_project(df)

    # Create the CFD chart
    cfd_chart = create_cfd_chart(avg_ratio_by_project)



    # First, filter the data for the current sprint
    current_sprint_number = get_last_sprint_number(df)  # Assuming you have a function to get the current sprint number
    # Replace NaN values with an empty string before checking for containment
    df['Sprint'] = df['Sprint'].astype(str)
    df_current_sprint = df[df['Sprint'].str.contains(str(current_sprint_number))]

        # Calculate assignee capacity
    assignee_capacity = assignee_median_capacity(df_current_sprint, df_current_sprint['Avg_Ratio'])
    df_current_sprint['Assignee Capacity'] = assignee_capacity

    # Calculate workload as sum of story points for the filtered data
    workload = df_current_sprint['Story Points'].sum()

    # Calculate Assignee capacity in the current sprint after deducting the workload
    assignee_capacity_current_sprint = assignee_capacity - workload

    # Create a DataFrame to hold the values for the horizontal bar chart
    data = pd.DataFrame({
        'Assignee': df_current_sprint['Assignee'],
        'Workload': workload,
        'Capacity': assignee_capacity_current_sprint
    })

    # Normalize the capacity and workload to percentages (of 100)
    total_capacity = assignee_capacity
    data['Assignee Workload'] = (data['Workload'] / total_capacity) * 100
    data['Assignee Capacity'] = (data['Capacity'] / total_capacity) * 100

    # Create the horizontal bar chart
    assignee_capacity_fig = px.bar(
        data,
        x=['Assignee Capacity', 'Assignee Workload'],
        y='Assignee',
        orientation='h',
        barmode='relative',  # To show both workload and capacity as percentages of total capacity
        title=f'Assignee Capacity and Workload in Sprint {current_sprint_number}',
        labels={'value': 'Percentage', 'variable': 'Metric'},
        color_discrete_map={'Assignee Capacity': 'green', 'Assignee Workload': 'red'}
    )

    # Set the range of the X-axis to [0, 100]
    assignee_capacity_fig.update_xaxes(range=[0, 100])

    # Initialize meeting_summary to an empty DataFrame
    meeting_summary = pd.DataFrame()

    logging.basicConfig(level=logging.INFO)


    if meetings is not None and not meetings.empty:
        logging.info("Processing meetings data...")
        # rest of the code...
    else:
        logging.info("No meetings data provided.")



    if meetings is not None and not meetings.empty:
        meeting_summary = meetings.groupby('Sprint').agg({'days': 'sum', 'Story Points': 'sum'}).reset_index()

        # Initialize meeting_fig to an empty chart first
        meeting_fig = go.Figure()
     
        # Create bar chart
        meeting_fig.add_trace(go.Bar(
            x=meeting_summary['Sprint'],
            y=meeting_summary['days'] * 8,  # Convert days to hours
            name='Hours Spent',
            marker_color='blue'
        ))





    # Display the charts in two columns
    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(fig2, use_container_width=True)
        st.subheader("Sprint Retrospective Word Cloud")
        word_cloud_data = generate_word_cloud_from_file("retro.txt")
        st.image(word_cloud_data)
        st.dataframe(meeting_summary)



    with col2:
        combined_chart = create_combined_chart(df, sprint_summary, sprint_totals)
        st.plotly_chart(combined_chart, use_container_width=True)
        st.plotly_chart(cfd_chart, use_container_width=True)
        #st.plotly_chart(meeting_fig, use_container_width=True)


    # Add a search input for the table
    search_value = st.text_input("Search for value in table rows:", "", key="search_input_tab2")

    # Filter the DataFrame based on the search input
    if search_value:
        filtered_df = df[df.apply(lambda row: search_value.lower() in str(row).lower(), axis=1)]
    else:
        filtered_df = df  # If no search input, show the original DataFrame

    # Display the filtered DataFrame as a table
    st.dataframe(filtered_df)

#-------------------------------------------------------------------------------------------------------------------------------------#



def display_tab3(df, assignee_rates):
    
    def custom_container(content, title=None):
        """Generates an HTML representation of a container mimicking mui.Paper."""
        style = """
            border: 1px solid #e0e0e0;
            padding: 16px;
            border-radius: 4px;
            resize: both;
            overflow: auto;
            margin: 8px 0;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        """
        header = f"<h4>{title}</h4>" if title else ""
        return f"""
            <div style="{style}">
                {header}
                {content}
            </div>
        """

    # Calculate the top assignees by ticket count
    top_assignees_table = df['Assignee'].value_counts().head(10).reset_index()
    top_assignees_table.columns = ['Assignee', 'Ticket Count']

    # Calculate the top clients by ticket count
    top_clients_table = df['CoreTimeClient'].value_counts().head(10).reset_index()
    top_clients_table.columns = ['CoreTimeClient', 'Ticket Count']

    # Create a scatter plot: Spent days vs. Delivered Story Points by Project
    project_data = df.groupby('CoreTimeProject').agg({'Hours': 'sum', 'days': 'sum', 'Story Points': 'sum'}).reset_index()
    Spent_days_Delivered_SPs = px.scatter(
        project_data,
        x='days',
        y='Story Points',
        color='CoreTimeProject',
        labels={'days': 'Spent days', 'Story Points': 'Delivered Story Points'},
        hover_data=['Hours'],
        title='Spent days vs. Delivered Story Points by Project'
    )

    # Create a scatter plot: Story Points by Project, Assignee, and Sprint
    df_grouped = df.groupby(['Project name', 'Assignee', 'Sprint'])['Story Points'].sum().reset_index()
    Projects_Assignees_Sprints = px.scatter(
        df_grouped,
        x='Project name',
        y='Assignee',
        size='Story Points',
        color='Sprint',
        labels={'Project name': 'Project Name', 'Assignee': 'Assignee', 'Sprint': 'Sprint', 'Story Points': 'Story Points'},
        size_max=60,
        title='Story Points by Project, Assignee, and Sprint'
    )

        # Display the charts in two columns
    col1, col2 = st.columns(2)

    with col1:
        st.table(top_assignees_table)
        st.plotly_chart(Projects_Assignees_Sprints, use_container_width=True)


    with col2:
        st.table(top_clients_table)
        st.plotly_chart(Spent_days_Delivered_SPs, use_container_width=True)

    # Add a search input for the table
    search_value = st.text_input("Search for value in table rows:", "", key="search_input_tab2")

    # Filter the DataFrame based on the search input
    if search_value:
        filtered_df = df[df.apply(lambda row: search_value.lower() in str(row).lower(), axis=1)]
    else:
        filtered_df = df  # If no search input, show the original DataFrame

    # Display the filtered DataFrame as a table
    st.dataframe(filtered_df)        



#-------------------------------------------------------------------------------------------------------------------------------------#
def get_last_sprint_number(df):
    # Ensure that all sprint values are strings and then extract the numbers
    sprint_numbers = [int(re.search(r'\d+', str(sprint)).group()) for sprint in df['Sprint'] if re.search(r'\d+', str(sprint))]

    if not sprint_numbers:  # Check if the list is empty
        st.warning("No valid sprint numbers found in the dataset.")
        return None

    return max(sprint_numbers)

#-------------------------------------------------------------------------------------------------------------------------------------#

def assignee_median_capacity(df, assignee):
    # Filter the dataframe for the specific assignee
    assignee_data = df[df['Assignee'] == assignee]

    # Group by sprint and sum the story points for each sprint
    total_story_points_per_sprint = assignee_data.groupby('Sprint')['Story Points'].sum()

    # Return the median of the summed story points across sprints
    return total_story_points_per_sprint.median()



def display_tab4(df, assignee_rates):

    # Create a radar chart for average ratio by assignee and core time phase
    avg_ratio_assignee_phase = df.groupby(['Assignee', 'CoreTimePhase'])['Avg_Ratio'].mean().reset_index()
    radar_chart_assignee_phase = px.line_polar(avg_ratio_assignee_phase, r='Avg_Ratio', theta='CoreTimePhase',
                                               line_close=True,
                                               title='Average Ratio by Assignee and Core Time Phase',
                                               labels={'Avg_Ratio': 'Average Ratio', 'CoreTimePhase': 'Core Time Phase'},
                                               color='Assignee')



    # Create a new column 'Assignee Capacity' to store the median story points per sprint for each assignee
    df['Assignee Capacity'] = df['Assignee'].apply(lambda x: assignee_median_capacity(df, x))

    assignee_capacity_fig = px.box(
        df,
        x='Assignee',
        y='Assignee Capacity',
        title='Assignee Capacity in Sprint'
    )

    # Add data labels to the box plot for individual data points only
    assignee_capacity_fig.update_traces(
        boxpoints='all',  # Display all the points
        jitter=0.3,
        pointpos=-1.8,
        hovertemplate='Capacity: %{y}<br><extra></extra>'
    )

    # Set the maximum value of the Y-axis
    assignee_capacity_fig.update_yaxes(range=[0, df['Assignee Capacity'].max() + 10])


    # Create average ratio bar chart
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

    # Line chart of Average Ratio by Sprint and Assignee
    line_chart_avg_ratio = px.line(
        df.groupby(['Sprint', 'Assignee'])['Avg_Ratio'].mean().reset_index(),
        x='Sprint',
        y='Avg_Ratio',
        color='Assignee',
        title='Average Ratio by Sprint and Assignee'
    )

    # Outer container
    outer_container = st.container()

    # Container 2: Charts
    with outer_container:
        container2 = st.container()
        col2, col3 = container2.columns(2)

        # Column 2
        col2.plotly_chart(avg_ratio_chart, use_container_width=True)
        col2.plotly_chart(radar_chart_assignee_phase, use_container_width=True)

        # Column 3
        col3.plotly_chart(assignee_capacity_fig, use_container_width=True)
        col3.plotly_chart(line_chart_avg_ratio, use_container_width=True)

    # Search input for the table
    search_value = st.text_input("Search for value in table rows:", "", key="search_input_tab4")

    # Filter the DataFrame based on the search input
    if search_value:
        filtered_df = df[df.apply(lambda row: search_value.lower() in str(row).lower(), axis=1)]
    else:
        filtered_df = df

    # Display the filtered DataFrame as a table
    st.dataframe(filtered_df)


#----------------------------------------------------------------------------------------#
def calculate_assignee_capacity(df, avg_ratio, sprint_duration_weeks=3, planning_days=1, working_days_per_week=5):
    # Calculate the number of development days
    dev_days = (sprint_duration_weeks * working_days_per_week) - planning_days
    
    total_story_points_delivered = df['Story Points'].sum()
    
    # Calculate average daily story point rate using only development days
    #avg_daily_story_point_rate = total_story_points_delivered / dev_days

    # Capacity is calculated using the total number of working days, including planning
    total_working_days = (sprint_duration_weeks * dev_days)
    assignee_capacity = avg_ratio * dev_days

    return assignee_capacity


def display_tab5(df, assignee_rates, sprint_bins):
    # Remove whitespace from column names (if any)
    df.columns = df.columns.str.strip()

    # First, filter the data for the current sprint
    current_sprint_number = get_last_sprint_number(df)  # Assuming you have a function to get the current sprint number
    
    # Handle NaN values when filtering by string contains
    mask = df['Sprint'].astype(str).str.contains(str(current_sprint_number)).fillna(False)
    df_current_sprint = df[mask]

    # Assuming the calculate_assignee_capacity function is defined elsewhere
    assignee_capacity = calculate_assignee_capacity(df_current_sprint, df_current_sprint['Avg_Ratio'])
    df_current_sprint['Assignee Capacity'] = assignee_capacity

    workload = df_current_sprint['Story Points'].sum()
    assignee_capacity_current_sprint = assignee_capacity - workload

    data = pd.DataFrame({
        'Assignee': df_current_sprint['Assignee'],
        'Workload': workload,
        'Capacity': assignee_capacity_current_sprint
    })

    total_capacity = assignee_capacity
    data['Assignee Workload'] = (data['Workload'] / total_capacity) * 100
    data['Assignee Capacity'] = (data['Capacity'] / total_capacity) * 100

    assignee_capacity_fig = px.bar(
        data,
        x=['Assignee Capacity', 'Assignee Workload'],
        y='Assignee',
        orientation='h',
        barmode='relative',
        title=f'Assignee Capacity and Workload in Sprint {current_sprint_number}',
        labels={'value': 'Percentage', 'variable': 'Metric'},
        color_discrete_map={'Assignee Capacity': 'green', 'Assignee Workload': 'red'}
    )
    assignee_capacity_fig.update_xaxes(range=[0, 100])
    assignee_capacity_fig.update_traces(texttemplate='%{value:.1f}', textposition='inside')

    # Convert datetime columns
    df_current_sprint['Created'] = pd.to_datetime(df_current_sprint['Created'], format='%d/%m/%Y %H:%M')
    df_current_sprint['Resolved'] = pd.to_datetime(df_current_sprint['Resolved'], format='%d/%m/%Y %H:%M')

    # Define the start and end dates for the current sprint
    start_date = pd.to_datetime("02/Aug/23 2:03 PM", format='%d/%b/%y %I:%M %p')
    end_date = pd.to_datetime("23/Aug/23 2:03 PM", format='%d/%b/%y %I:%M %p')

    # Filter tasks that were both created and resolved within the current sprint
    df_current_sprint_filtered = df_current_sprint[
        (df_current_sprint['Created'] >= start_date) & 
        (df_current_sprint['Resolved'] <= end_date)
    ]

    # Calculate the 'Resolution Time' column for the filtered tasks
    df_current_sprint = df_current_sprint.dropna(subset=['Resolved'])

    df_current_sprint_filtered['Resolution Time'] = (df_current_sprint_filtered['Resolved'] - df_current_sprint_filtered['Created']).dt.days


    # Aggregate the resolution time by assignee (use the filtered data)
    resolution_by_assignee = df_current_sprint_filtered.groupby('Assignee')['Resolution Time'].sum().reset_index()

    resolution_time_fig = px.bar(
        resolution_by_assignee,
        x='Assignee',
        y='Resolution Time',
        title=f'Under construction - Resolution Time per Assignee in Sprint {current_sprint_number}',
        labels={'Resolution Time': 'Total Resolution Time (days)'},
        orientation='v'
    )
    resolution_time_fig.update_traces(texttemplate='%{value}', textposition='outside')

        # Define the start date for the current sprint
    start_date = pd.to_datetime("02/Aug/23 2:03 PM", format='%d/%b/%y %I:%M %p')

    total_story_points_current_sprint = df_current_sprint['Story Points'].sum()

    df_after_date = df_current_sprint[df_current_sprint['Created'] > start_date]
    total_story_points_after_date = df_after_date['Story Points'].sum()

    data_added_to_sprint = pd.DataFrame({
        'Category': ['Total in Current Sprint', f'Added After {start_date.strftime("%d/%b/%Y %I:%M %p")}'],
        'Story Points': [total_story_points_current_sprint, total_story_points_after_date]
    })

    added_to_sprint_fig = px.bar(
        data_added_to_sprint,
        x='Category',
        y='Story Points',
        title=f'Story Points Added to Sprint {current_sprint_number}',
        labels={'Category': 'Category', 'Story Points': 'Story Points'},
        orientation='v'
    )
    added_to_sprint_fig.update_traces(texttemplate='%{value}', textposition='outside')



    grouped_df = df_current_sprint.groupby(['Assignee', 'Status'])['Story Points'].sum().reset_index()
    
    status_assignee_fig = px.bar(
        grouped_df,
        x='Assignee',
        y='Story Points',
        color='Status',
        title=f'Story Points by Status and Assignee in Sprint {current_sprint_number}',
        labels={'Story Points': 'Story Points'},
        orientation='v'
    )
    status_assignee_fig.update_traces(texttemplate='%{value}', textposition='outside')

    # Filter the DataFrame for the current sprint and then group by CoreTimeClient
    df_current_sprint_hours = df[df['Sprint'].str.contains(str(current_sprint_number))]
    hours_by_client = df_current_sprint_hours.groupby('CoreTimeClient')['Hours'].sum().reset_index()


    # Create the 3D pie chart
    fig = go.Figure(data=[go.Pie(
        labels=hours_by_client['CoreTimeClient'],
        values=hours_by_client['Hours'],
        hovertemplate="%{label}: %{value} hours",
        textinfo='label+percent',
        pull=[0.1] * len(hours_by_client)  # Set the distance of each sector from the center
    )])

    fig.update_layout(
        title="Worked Hours in the Current SprintS by CoreTimeClient",
        scene=dict(
            aspectratio=dict(x=1, y=1, z=0.7),
            camera_eye=dict(x=1.2, y=1.2, z=0.6),
            dragmode="turntable",
        ),
        showlegend=False  # Hide the legend for a cleaner look
    )


    
    # Display the charts in two columns
    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(assignee_capacity_fig, use_container_width=True)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.plotly_chart(added_to_sprint_fig, use_container_width=True)
        st.plotly_chart(status_assignee_fig, use_container_width=True)

    search_value = st.text_input("Search for value in table rows:", "", key="search_input_tab2")

    # Filter the DataFrame based on the search input
    if search_value:
        filtered_df = df[df.apply(lambda row: search_value.lower() in str(row).lower(), axis=1)]
    else:
        filtered_df = df  # If no search input, show the original DataFrame

    # Display the filtered DataFrame as a table
    st.dataframe(filtered_df)




def display_Ad_Hoc_Analysis(df, assignee_rates):
    df['Work date'] = pd.to_datetime(df['Work date'], infer_datetime_format=True)
    # Assignee rates are already passed as an argument, no need to call get_assignee_rates again here
    df['Cost'] = df['Hours'] * df['Assignee'].map(assignee_rates)

    treemap_data = df.groupby(['CoreTimeClient', 'CoreTimeProject', 'CoreTimePhase', 'CoreTimeActivity', pd.Grouper(key='Work date', freq='M')])['Cost'].sum().reset_index()

    treemap_fig = px.treemap(
        treemap_data,
        path=['Work date', 'CoreTimeClient', 'CoreTimeProject', 'CoreTimePhase', 'CoreTimeActivity'],
        values='Cost',
        title='Cost by CoreTimeClient, Project, Phase, Activity, and Month'
    )

    treemap_fig.update_layout(
        height=600,
        width=1000,
        margin=dict(l=0, r=0, t=50, b=0),
        plot_bgcolor='rgba(0, 0, 0, 0)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        shapes=[
            dict(
                type='rect',
                xref='paper',
                yref='paper',
                x0=0,
                y0=0,
                x1=1,
                y1=1,
                line=dict(color='black', width=1),
                fillcolor='rgba(0, 0, 0, 0)'
            )
        ]
    )

    # Add new chart: Line chart of Delivered Story Points by Sprint and Assignee
    line_chart_delivered_sp = px.line(
        df.groupby(['Sprint', 'Assignee'])['Story Points'].sum().reset_index(),
        x='Sprint',
        y='Story Points',
        color='Assignee',
        labels={'Sprint': 'Sprint', 'Story Points': 'Delivered Story Points', 'Assignee': 'Assignee'},
        title='Delivered Story Points by Sprint and Assignee'
    )

    # Add new chart: Line chart of Average Ratio by Sprint and Assignee
    line_chart_avg_ratio = px.line(
        df.groupby(['Sprint', 'Assignee'])['Avg_Ratio'].mean().reset_index(),
        x='Sprint',
        y='Avg_Ratio',
        color='Assignee',
        labels={'Sprint': 'Sprint', 'Avg_Ratio': 'Average Ratio', 'Assignee': 'Assignee'},
        title='Average Ratio by Sprint and Assignee'
    )

    # Add new chart: Line chart of Time Booked by Sprint and Assignee
    line_chart_time_booked = px.line(
        df.groupby(['Sprint', 'Assignee'])['Hours'].sum().reset_index(),
        x='Sprint',
        y='Hours',
        color='Assignee',
        labels={'Sprint': 'Sprint', 'Hours': 'Time Booked', 'Assignee': 'Assignee'},
        title='Time Booked by Sprint and Assignee'
    )

    # Add new chart: Line chart of Delivered Story Points vs Booked Hours by Sprint for Aleksander 
    aleksander_data = df[df['Assignee'] == 'Aleksander ']
    line_chart_delivered_vs_hours = px.line(
        aleksander_data.groupby('Sprint')[['Story Points', 'Hours']].sum().reset_index(),
        x='Sprint',
        y=['Story Points', 'Hours'],
        labels={'Sprint': 'Sprint', 'value': 'Value', 'variable': 'Metric'},
        title='Delivered Story Points vs Booked Hours by Sprint (Aleksander )'
    )

    # Add new chart: Line chart of Delivered Story Points and Average Ratio by Sprint for Aleksander 
    line_chart_delivered_and_ratio = px.line(
        aleksander_data.groupby('Sprint').agg({'Story Points': 'sum', 'Avg_Ratio': 'mean'}).reset_index(),
        x='Sprint',
        y=['Story Points', 'Avg_Ratio'],
        labels={'Sprint': 'Sprint', 'value': 'Value', 'variable': 'Metric'},
        title='Delivered Story Points and Average Ratio by Sprint (Aleksander )'
    )

    #st.plotly_chart(line_chart_delivered_sp, line_chart_avg_ratio, line_chart_time_booked, line_chart_delivered_vs_hours, line_chart_delivered_and_ratio, treemap_fig)
    #st.plotly_chart(line_chart_time_booked, line_chart_delivered_vs_hours, line_chart_delivered_and_ratio, treemap_fig)

DEFAULT_RATES = {
    "Pawel G": 42.80,
    "Marcin Ko": 42.80,
    "Marcin Kl": 26.75,
    "Lukasz": 31.03,
    "Alek D": 26.75,
    "Dawid N": 26.75
}


#---------------------------------------------------------------------------------------------------#
def run_app():
    st.set_page_config(layout='wide')

    # Tabs at the top of the sidebar
    tabs = {
        #"Current Sprint": display_tab5,
        "Team Performance": display_tab2,
        "Individual Performance": display_tab4,
        "Costs": display_tab1,
        "Productivity & Workload": display_tab3,
        "Ad Hoc Analysis": display_Ad_Hoc_Analysis
    }
    
    selected_tab = st.sidebar.radio("Select a Tab", list(tabs.keys()))
    if 'selected_tab' not in st.session_state or st.session_state.selected_tab != selected_tab:
        st.session_state.selected_tab = selected_tab  # Store the selected tab in session state

    # Place the file uploaders at the bottom of the sidebar
    with st.sidebar.expander("Upload Files"):
        uploaded_file_iterative = st.file_uploader("Choose Iterative CSV file", type="csv")
        uploaded_file_eigen = st.file_uploader("Choose Eigen CSV file", type="csv")

    # If both files are uploaded, process them
    df = None
    if uploaded_file_iterative and uploaded_file_eigen:
        df, meetings = load_data(uploaded_file_iterative, uploaded_file_eigen, sprint_bins)


    if df is not None:
        # Get the last sprint number
        last_sprint_number = get_last_sprint_number(df)
        st.title(f"Dev Sprint 80 - {last_sprint_number}")

        # Collapsible section for assignee rates
        with st.expander("Assignee Rates"):
            assignee_rates = get_assignee_rates(df)  # Retrieve assignee rates

        # Call the appropriate function for the selected tab
        selected_function = tabs[selected_tab]
        selected_function(df, assignee_rates)  # Pass assignee_rates as a parameter
    else:
        st.warning("Upload files.")

if __name__ == "__main__":
    run_app()