import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(layout="wide")

data = pd.read_csv("gym_members_exercise_tracking.csv")

# CSS to center the elements
st.markdown(
    """
    <style>
    .center {
        display: flex;
        justify-content: center;
        text-align: center;
        color: #636EFA
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Centering the headers
st.markdown("<h1 class='center'>An EsteStyle Streamlit Page Where Python Wiz Meets Data Viz!</h1>", unsafe_allow_html=True)
st.markdown("<h1 class='center'></h1>", unsafe_allow_html=True)

st.markdown("<img src='https://1drv.ms/i/s!ArWyPNkF5S-foZspwsary83MhqEWiA?embed=1&width=307&height=307' width='300' style='display: block; margin: 0 auto;'>" , unsafe_allow_html=True)

st.markdown("<h1 class='center'> </h1>", unsafe_allow_html=True)

st.markdown("<h1 class='center'>Gym Members Exercise Dataset</h1>", unsafe_allow_html=True)

col_1, col_2= st.columns([1, 1], gap="large")

with col_1:
    st.title("Data Definitions")
    st.write("**Age**: *Age of the gym member*")
    st.write("**Gender**: *Gender of the gym member (Male or Female)*")
    st.write("**Weight (kg)**: *Member’s weight in kilograms*")
    st.write("**Height (m)**: *Member’s height in meters*")
    st.write("**Max_BPM**: *Maximum heart rate (beats per minute) during workout sessions*")
    st.write("**Avg_BPM**: *Average heart rate during workout sessions*")
    st.write("**Resting_BPM**: *Heart rate at rest before workout*")
    st.write("**Session_Duration (hours)**: *Duration of each workout session in hours*")
    st.write("**Calories_Burned**: *Total calories burned during each session*")
    st.write("**Workout_Type**: *Type of workout performed (e.g., Cardio, Strength, Yoga, HIIT)*")
    st.write("**Fat_Percentage**: *Body fat percentage of the member*")
    st.write("**Water_Intake (liters)**: *Daily water intake during workouts*")
    st.write("**Workout_Frequency (days/week)**: *Number of workout sessions per week*")
    st.write("**Experience_Level**: *Level of experience, from beginner (1) to expert (3)*")
    st.write("**BMI**: *Body Mass Index, calculated from height and weight*")
    st.write("")

st.title("Exploritory Data Analysis")
st.write("Data category disribution visualizations")

with col_2:
    st.title("Original Dataset")
    st.dataframe(data)

    st.markdown("""
    ### Data Attribution
    Dataset: ["Gym Members Exercise Dataset"](https://www.kaggle.com/datasets/valakhorasani/gym-members-exercise-dataset) by Vala Khorasani, accessed on November 5, 2024.
    """)

col1, col2, col3 = st.columns([1, 1, 1], gap="large")

with col1:
    
    # Drop missing values for 'Avg_BPM' and store in a variable
    age_data = data['Age'].dropna()

    # Create a Plotly histogram
    fig = go.Figure(
        data=go.Histogram(
            x=age_data,
            nbinsx=42,  # Set number of bins
            marker=dict(color='#636EFA', line=dict(color='black', width=1)),  # Bin color and edge color
        )
    )

    # Update layout
    fig.update_layout(
        title="Age",
        xaxis_title="Age",
        yaxis_title="Frequency",
        plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
    )

    # Display the Plotly figure in Streamlit
    st.plotly_chart(fig)

    # Drop missing values for 'Avg_BPM' and store in a variable
    mbpm_data = data['Max_BPM'].dropna()

    # Create a Plotly histogram
    fig = go.Figure(
        data=go.Histogram(
            x=mbpm_data,
            nbinsx=30,  # Set number of bins
            marker=dict(color='#636EFA', line=dict(color='black', width=1)),  # Bin color and edge color
        )
    )

    # Update layout
    fig.update_layout(
        title="Max Heart Beats Per Min",
        xaxis_title="BPM",
        yaxis_title="Frequency",
        plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
    )

    # Display the Plotly figure in Streamlit
    st.plotly_chart(fig)

    # Drop missing values for 'Avg_BPM' and store in a variable
    height_data = data['Height (m)'].dropna()

    # Create a Plotly histogram
    fig = go.Figure(
        data=go.Histogram(
            x=height_data,
            nbinsx=30,  # Set number of bins
            marker=dict(color='#636EFA', line=dict(color='black', width=1)),  # Bin color and edge color
        )
    )

    # Update layout
    fig.update_layout(
        title="Height (m)",
        xaxis_title="Height",
        yaxis_title="Frequency",
        plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
    )

    # Display the Plotly figure in Streamlit
    st.plotly_chart(fig)

    
    # Drop missing values for 'Avg_BPM' and store in a variable
    sess_data = data['Session_Duration (hours)'].dropna()

    # Create a Plotly histogram
    fig = go.Figure(
        data=go.Histogram(
            x=sess_data,
            nbinsx=30,  # Set number of bins
            marker=dict(color='#636EFA', line=dict(color='black', width=1)),  # Bin color and edge color
        )
    )

    # Update layout
    fig.update_layout(
        title="Session Duration (hr)",
        xaxis_title="Duration",
        yaxis_title="Frequency",
        plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
    )

    # Display the Plotly figure in Streamlit
    st.plotly_chart(fig)

    freq_counts = data['Workout_Frequency (days/week)'].value_counts()

    # Create a bar chart using Plotly
    fig = go.Figure(
        data=go.Bar(
            x=freq_counts.index,
            y=freq_counts.values,
            marker=dict(color=['#AB63FA', '#FFA15A', '#19D3F3', '#FF6692']),
            text=freq_counts.values,  # Show counts on bars
            textposition='outside'  # Position text outside bars for clarity
        )
    )

    # Update layout
    fig.update_layout(
        title="Workout Frequence",
        xaxis_title="Frequency",
        yaxis_title="Count",
        xaxis=dict(tickmode='array', tickvals=freq_counts.index),
        yaxis=dict(showline=True, linewidth=1, linecolor='black'),
        plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
    )

    # Display in Streamlit
    st.plotly_chart(fig)


with col2:

    # Drop missing values for 'Avg_BPM' and store in a variable
    fat_data = data['Fat_Percentage'].dropna()

    # Create a Plotly histogram
    fig = go.Figure(
        data=go.Histogram(
            x=fat_data,
            nbinsx=30,  # Set number of bins
            marker=dict(color='#636EFA', line=dict(color='black', width=1)),  # Bin color and edge color
        )
    )

    # Update layout
    fig.update_layout(
        title="Body Fat Percentage",
        xaxis_title="Fat %",
        yaxis_title="Frequency",
        plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
    )

    # Display the Plotly figure in Streamlit
    st.plotly_chart(fig)

    # Drop missing values for 'Avg_BPM' and store in a variable
    rest_data = data['Resting_BPM'].dropna()

    # Create a Plotly histogram
    fig = go.Figure(
        data=go.Histogram(
            x=rest_data,
            nbinsx=25,  # Set number of bins
            marker=dict(color='#636EFA', line=dict(color='black', width=1)),  # Bin color and edge color
        )
    )

    # Update layout
    fig.update_layout(
        title="Resting Heart Beats Per Min",
        xaxis_title="BPM",
        yaxis_title="Frequency",
        plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
    )

    # Display the Plotly figure in Streamlit
    st.plotly_chart(fig)

    # Drop missing values for 'Avg_BPM' and store in a variable
    weight_data = data['Weight (kg)'].dropna()

    # Create a Plotly histogram
    fig = go.Figure(
        data=go.Histogram(
            x=weight_data,
            nbinsx=25,  # Set number of bins
            marker=dict(color='#636EFA', line=dict(color='black', width=1)),  # Bin color and edge color
        )
    )

    # Update layout
    fig.update_layout(
        title="Weight (kg)",
        xaxis_title="Calories Burned",
        yaxis_title="Frequency",
        plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
    )

    # Display the Plotly figure in Streamlit
    st.plotly_chart(fig)
    

    # Drop missing values for 'Avg_BPM' and store in a variable
    burn_data = data['Calories_Burned'].dropna()

    # Create a Plotly histogram
    fig = go.Figure(
        data=go.Histogram(
            x=burn_data,
            nbinsx=25,  # Set number of bins
            marker=dict(color='#636EFA', line=dict(color='black', width=1)),  # Bin color and edge color
        )
    )

    # Update layout
    fig.update_layout(
        title="Calories Burned",
        xaxis_title="Calories",
        yaxis_title="Frequency",
        plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
    )

    # Display the Plotly figure in Streamlit
    st.plotly_chart(fig)

    type_counts = data['Workout_Type'].value_counts()

    # Create a bar chart using Plotly
    fig = go.Figure(
        data=go.Bar(
            x=type_counts.index,
            y=type_counts.values,
            marker=dict(color=['#AB63FA', '#FFA15A', '#19D3F3', '#FF6692']),
            text=type_counts.values,  # Show counts on bars
            textposition='outside'  # Position text outside bars for clarity
        )
    )

    # Update layout
    fig.update_layout(
        title="Workout Type",
        xaxis_title="Type",
        yaxis_title="Count",
        xaxis=dict(tickmode='array', tickvals=type_counts.index),
        yaxis=dict(showline=True, linewidth=1, linecolor='black'),
        plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
    )

    # Display in Streamlit
    st.plotly_chart(fig)

with col3: 

    gender_counts = data['Gender'].value_counts()

    # Create a bar chart using Plotly
    fig = go.Figure(
        data=go.Bar(
            x=gender_counts.index,
            y=gender_counts.values,
            marker=dict(color=['#40E0D0', '#636EFA']),  # Example colors for categories
            text=gender_counts.values,  # Show counts on bars
            textposition='outside'  # Position text outside bars for clarity
        )
    )

    # Update layout
    fig.update_layout(
        title="Gender",
        xaxis_title="Gender",
        yaxis_title="Count",
        xaxis=dict(tickmode='array', tickvals=gender_counts.index),
        yaxis=dict(showline=True, linewidth=1, linecolor='black'),
        plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
    )

    # Display the Plotly figure in Streamlit
    st.plotly_chart(fig)

    # Drop missing values for 'Avg_BPM' and store in a variable
    bpm_data = data['Avg_BPM'].dropna()

    # Create a Plotly histogram
    fig = go.Figure(
        data=go.Histogram(
            x=bpm_data,
            nbinsx=25,  # Set number of bins
            marker=dict(color='#636EFA', line=dict(color='black', width=1)),  # Bin color and edge color
        )
    )

    # Update layout
    fig.update_layout(
        title="Avg Heart Beats Per Min",
        xaxis_title="BPM",
        yaxis_title="Frequency",
        plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
    )

    # Display the Plotly figure in Streamlit
    st.plotly_chart(fig)

    # Drop missing values for 'Avg_BPM' and store in a variable
    bmi_data = data['BMI'].dropna()

    # Create a Plotly histogram
    fig = go.Figure(
        data=go.Histogram(
            x=bmi_data,
            nbinsx=25,  # Set number of bins
            marker=dict(color='#636EFA', line=dict(color='black', width=1)),  # Bin color and edge color
        )
    )

    # Update layout
    fig.update_layout(
        title="Body Mass Index",
        xaxis_title="BMI",
        yaxis_title="Frequency",
        plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
    )

    # Display the Plotly figure in Streamlit
    st.plotly_chart(fig)

    # Drop missing values for 'Avg_BPM' and store in a variable
    water_data = data['Water_Intake (liters)'].dropna()

    # Create a Plotly histogram
    fig = go.Figure(
        data=go.Histogram(
            x=water_data,
            nbinsx=25,  # Set number of bins
            marker=dict(color='#636EFA', line=dict(color='black', width=1)),  # Bin color and edge color
        )
    )

    # Update layout
    fig.update_layout(
        title="Water Intake (liters)",
        xaxis_title="Intake",
        yaxis_title="Frequency",
        plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
    )

    # Display the Plotly figure in Streamlit
    st.plotly_chart(fig)

    expr_counts = data['Experience_Level'].value_counts()

    # Create a bar chart using Plotly
    fig = go.Figure(
        data=go.Bar(
            x=expr_counts.index,
            y=expr_counts.values,
            marker=dict(color=['#AB63FA', '#FFA15A', '#19D3F3']),
            text=expr_counts.values,  # Show counts on bars
            textposition='outside'  # Position text outside bars for clarity
        )
    )

    # Update layout
    fig.update_layout(
        title="Experience Level",
        xaxis_title="Level",
        yaxis_title="Count",
        xaxis=dict(tickmode='array', tickvals=expr_counts.index),
        yaxis=dict(showline=True, linewidth=1, linecolor='black'),
        plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
    )

    # Display in Streamlit
    st.plotly_chart(fig)

# Select only numeric columns
numeric_data = data.select_dtypes(include=['number'])
corr_matrix = numeric_data.corr()

# Create the Plotly heatmap
fig = go.Figure(
    data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='balance',
        colorbar=dict(title="Correlation")
    )
)

# Add text annotations (correlation values) on top of the heatmap
for i in range(len(corr_matrix.columns)):
    for j in range(len(corr_matrix.columns)):
        fig.add_annotation(
            x=corr_matrix.columns[i],
            y=corr_matrix.columns[j],
            text=f"{corr_matrix.values[i, j]:.1f}",
            showarrow=False,
            font=dict(color="black" if abs(corr_matrix.values[i, j]) < 0.5 else "white")
        )

# Update layout for readability
fig.update_layout(
    title="Correlation Heatmap",
    xaxis=dict(tickmode='array', tickvals=corr_matrix.columns),
    yaxis=dict(tickmode='array', tickvals=corr_matrix.columns)
)

# Display in Streamlit
st.plotly_chart(fig)

st.header("Data distributions analysis and conclusions:")
st.write("""
         
         - Age and Gender categories aren't skewed enabling comparitive analysis
         
         - Most data categories are noramlly distributed allowing for greater popultion generalizations
         
         - Correlations between numeric categories indicate the presence of variable relationships
    
         """)

st.title("DataFrame cleaning and manipulations")
st.write("Manipulated the dataframe to apply unit conversions.")
st.write("Dropped any row of data if value(s) within its column were 2.5 std deviations from the column's mean to eliminate outliers.")

code = '''
# Convert and store m to ft and kg to lbs in new columns
data['Height (ft)'] = data['Height (m)'] * 3.28084
data['Weight (lbs)'] = data['Weight (kg)'] * 2.20462

# Drop the original meter and kilogram columns
data.drop(columns=['Height (m)', 'Weight (kg)'], inplace=True)

# Create DataFrame
df = pd.DataFrame(data)

# Select only numeric columns
numeric_cols = df.select_dtypes(include=np.number).columns

# Calculate z-scores for numeric columns
z_scores = df[numeric_cols].apply(lambda x: (x - x.mean()) / x.std())

# Filter rows where all numeric column values are within 2.5 standard deviations
filtered_df = df[(np.abs(z_scores) <= 2.5).all(axis=1)]
'''

st.code(code, language='python')

# Convert and store m to ft and kg to lbs in new columns
data['Height (ft)'] = data['Height (m)'] * 3.28084
data['Weight (lbs)'] = data['Weight (kg)'] * 2.20462

# Drop the original meter and kilogram columns
data.drop(columns=['Height (m)', 'Weight (kg)'], inplace=True)

# Create DataFrame
df = pd.DataFrame(data)

# Select only numeric columns
numeric_cols = df.select_dtypes(include=np.number).columns

# Calculate z-scores for numeric columns
z_scores = df[numeric_cols].apply(lambda x: (x - x.mean()) / x.std())

# Filter rows where all numeric column values are within 2.5 standard deviations
filtered_df = df[(np.abs(z_scores) <= 2.5).all(axis=1)]

st.title("Physical Metric Changes Over Time and Between Genders")

# Define non-overlapping bins
bins = list(range(18, 59, 5))  # Start at 18, end at 63, step by 7 years
age_df = filtered_df.copy()
age_df['age_bins'] = pd.cut(age_df['Age'], bins=bins, right=False)

# Calculate mean and SD for each age group
group_stats = age_df.groupby(('age_bins'), observed=False)['Weight (lbs)'].agg(['mean', 'std'])
means = group_stats['mean']
errors = group_stats['std']

# Convert age_bins to string for better display
age_bins_str = [f"{int(interval.left)}-{int(interval.right - 1)}" for interval in means.index]

# Create the bar chart with Plotly
colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880']
fig = go.Figure(data=[
    go.Bar(
        x=age_bins_str,
        y=means,
        error_y=dict(
            type='data',
            array=errors,
            visible=False
        ),
        marker=dict(color=colors[:len(means)])
    )
])

# Update layout for titles and labels
fig.update_layout(
    title="Average Weight by Age Group",
    xaxis_title="Age Group (5-year bins)",
    yaxis_title="Average Weight (lbs)",
    yaxis=dict(range=[145, means.max() + 5])  # Adjust as needed
)

# Display the Plotly figure in Streamlit
st.plotly_chart(fig)

# Group the data by age_bins and gender and calculate mean and std for Fat_Percentage
group_stats = age_df.groupby(['age_bins', 'Gender'], observed=False)['Weight (lbs)'].agg(['mean', 'std']).reset_index()

# Prepare lists for the bar chart
genders = group_stats['Gender'].unique()
fig = go.Figure()

color_map = {
    'Male': '#40E0D0',  # Turquoise
    'Female': '#636EFA'  # Teal
}

# Create bar traces for each gender
for gender in genders:
    gender_data = group_stats[group_stats['Gender'] == gender]
    fig.add_trace(
        go.Bar(
            x=age_bins_str,
            y=gender_data['mean'],
            name=gender,
            error_y=dict(
                type='data',
                array=gender_data['std'],
                visible=False
            ),
            marker=dict(color=color_map[gender])
        )
    )

# Update layout for titles and labels
fig.update_layout(
    title="Weight by Age Group and Gender",
    xaxis_title="Age Group (5-year bins)",
    yaxis_title="Weight (lbs)",
    yaxis=dict(range=[100, group_stats['mean'].max() + 20]),
    barmode='group'  # Group bars by age_bins
)

# Display the Plotly figure in Streamlit
st.plotly_chart(fig)

# Calculate mean and SD for each age group
group_stats = age_df.groupby(('age_bins'), observed=False)['BMI'].agg(['mean', 'std'])
means = group_stats['mean']
errors = group_stats['std']

# Convert age_bins to string for better display
age_bins_str = [f"{int(interval.left)}-{int(interval.right - 1)}" for interval in means.index]

# Create the bar chart with Plotly
colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880']
fig = go.Figure(data=[
    go.Bar(
        x=age_bins_str,
        y=means,
        error_y=dict(
            type='data',
            array=errors,
            visible=False
        ),
        marker=dict(color=colors[:len(means)])
    )
])

# Update layout for titles and labels
fig.update_layout(
    title="BMI by Age Group",
    xaxis_title="Age Group (5-year bins)",
    yaxis_title="BMI",
    yaxis=dict(range=[21, means.max() + 1])  # Adjust as needed
)

# Calculate mean and SD for each age group
group_stats = age_df.groupby(('age_bins'), observed=False)['BMI'].agg(['mean', 'std'])
means = group_stats['mean']
errors = group_stats['std']

# Display the Plotly figure in Streamlit
st.plotly_chart(fig)

# Group the data by age_bins and gender and calculate mean and std for Fat_Percentage
group_stats = age_df.groupby(['age_bins', 'Gender'], observed=False)['BMI'].agg(['mean', 'std']).reset_index()

# Prepare lists for the bar chart
genders = group_stats['Gender'].unique()
fig = go.Figure()

color_map = {
    'Male': '#40E0D0',  # Turquoise
    'Female': '#636EFA'  # Teal
}

# Create bar traces for each gender
for gender in genders:
    gender_data = group_stats[group_stats['Gender'] == gender]
    fig.add_trace(
        go.Bar(
            x=age_bins_str,
            y=gender_data['mean'],
            name=gender,
            error_y=dict(
                type='data',
                array=gender_data['std'],
                visible=False
            ),
            marker=dict(color=color_map[gender])
        )
    )

# Update layout for titles and labels
fig.update_layout(
    title="BMI by Age Group and Gender",
    xaxis_title="Age Group (5-year bins)",
    yaxis_title="BMI",
    yaxis=dict(range=[20, group_stats['mean'].max() + 1]),
    barmode='group'  # Group bars by age_bins
)

# Display the Plotly figure in Streamlit
st.plotly_chart(fig)

# Create the bar chart with Plotly
colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880']
fig = go.Figure(data=[
    go.Bar(
        x=age_bins_str,
        y=means,
        error_y=dict(
            type='data',
            array=errors,
            visible=False
        ),
        marker=dict(color=colors[:len(means)])
    )
])

# Update layout for titles and labels
fig.update_layout(
    title="Fat % by Age Group",
    xaxis_title="Age Group (5-year bins)",
    yaxis_title="Fat %",
    yaxis=dict(range=[21, means.max() + 1])  # Adjust as needed
)

# Display the Plotly figure in Streamlit
st.plotly_chart(fig)

# Group the data by age_bins and gender and calculate mean and std for Fat_Percentage
group_stats = age_df.groupby(['age_bins', 'Gender'], observed=False)['Fat_Percentage'].agg(['mean', 'std']).reset_index()

# Prepare lists for the bar chart
genders = group_stats['Gender'].unique()
fig = go.Figure()

color_map = {
    'Male': '#40E0D0',  # Turquoise
    'Female': '#636EFA'  # Teal
}

# Create bar traces for each gender
for gender in genders:
    gender_data = group_stats[group_stats['Gender'] == gender]
    fig.add_trace(
        go.Bar(
            x=age_bins_str,
            y=gender_data['mean'],
            name=gender,
            error_y=dict(
                type='data',
                array=gender_data['std'],
                visible=False
            ),
            marker=dict(color=color_map[gender])
        )
    )

# Update layout for titles and labels
fig.update_layout(
    title="Fat % by Age Group and Gender",
    xaxis_title="Age Group (5-year bins)",
    yaxis_title="Fat %",
    yaxis=dict(range=[21, group_stats['mean'].max() + 3]),
    barmode='group'  # Group bars by age_bins
)

# Display the Plotly figure in Streamlit
st.plotly_chart(fig)

st.header("Physical metric analysis and conclusions:")
st.write("""
         
         - Weight increases slightly between age ranges of 28-32 then decreases by age 38 only to once again increase 
         at year 48
         
         - These fluctuations in weight are observed in both genders until the later time periods 
         
         - The weight changes are mirrored by changes in BMI and within gender categories

         - **KEY FINDING**: The increase in weight in the later years appear to be driven by Male's increase in Fat Percentage, 
         while Female Fat Percentage decreases, explaining the downward trend in BMI observed during that time in Females
    
         """)

st.title("Positive Correlations and Trends Between Category Measures")
st.write(":red[r values displayed are Pearson correlation coefficients]")

col1, col2, col3 = st.columns([1, 1, 1], gap="large")

with col1:
    # Create a box plot using Plotly
    fig = go.Figure()

    # Add box traces for each workout frequency
    for workout_freq in filtered_df['Workout_Frequency (days/week)'].unique():
        fig.add_trace(
            go.Box(
                y=filtered_df[filtered_df['Workout_Frequency (days/week)'] == workout_freq]['Session_Duration (hours)'],
                name=str(workout_freq),  # Set x-axis category
                #boxpoints='all',  # Display all points
                #jitter=0.3,  # Add some jitter to the points for better visibility
                #pointpos=-1.8,  # Position of points in relation to box
                marker=dict(color='#FF6692')  # Set color
            )
        )

    # Add text annotation with r value
    fig.add_annotation( 
        x=('5.5'),  # Position on the plot
        y=('2.2'),  # Adjust this to position the text better
        text=f'r = 0.6',  # Display r value
        showarrow=False,
        font=dict(size=18, color='red'),
        align='right'
    )

    # Update layout
    fig.update_layout(
        title="Session Duration by Workout Frequency",
        xaxis_title="Workout Frequency (days/week)",
        yaxis_title="Session Duration (hours)",
        showlegend=False
    )

    # Display the Plotly figure in Streamlit
    st.plotly_chart(fig)

with col2:
    # Create a box plot using Plotly
    fig = go.Figure()

    # Add box traces for each workout frequency
    for workout_expr in filtered_df['Experience_Level'].unique():
        fig.add_trace(
            go.Box(
                y=filtered_df[filtered_df['Experience_Level'] == workout_expr]['Session_Duration (hours)'],
                name=str(workout_expr),  # Set x-axis category
                #boxpoints='all',  # Display all points
                #jitter=0.3,  # Add some jitter to the points for better visibility
                #pointpos=-1.8,  # Position of points in relation to box
                marker=dict(color='#FF6692')  # Set color
            )
        )

    # Add text annotation with r value
    fig.add_annotation( 
        x=('3.5'),  # Position on the plot
        y=('2.2'),  # Adjust this to position the text better
        text=f'r = 0.8',  # Display r value
        showarrow=False,
        font=dict(size=18, color='red'),
        align='right'
    )

    # Update layout
    fig.update_layout(
        title="Session Duration by Workout Experience Level",
        xaxis_title="Workout Experience Level",
        yaxis_title="Session Duration (hours)",
        showlegend=False
    )

    # Display the Plotly figure in Streamlit
    st.plotly_chart(fig)

with col3:
    # Group the data and calculate the mean or count of Experience_Level for each Workout_Frequency
    group_stats = filtered_df.groupby('Workout_Frequency (days/week)')['Experience_Level'].mean().reset_index()

    # Create a bar plot using Plotly
    fig = go.Figure()

    # Add bar trace
    fig.add_trace(
        go.Bar(
            x=group_stats['Workout_Frequency (days/week)'],
            y=group_stats['Experience_Level'],
            marker=dict(color='#FF6692')  # Set color
        )
    )

    # Add text annotation with r value
    fig.add_annotation( 
        x=('5.7'),  # Position on the plot
        y=('3.3'),  # Adjust this to position the text better
        text=f'r = 0.8',  # Display r value
        showarrow=False,
        font=dict(size=18, color='red'),
        align='right'
    )

    # Update layout
    fig.update_layout(
        title="Avg Workout Experience by Frequency",
        xaxis_title="Workout Frequency (days/week)",
        yaxis_title="Average Workout Experience Level",
        showlegend=False
    )

    # Display the Plotly figure in Streamlit
    st.plotly_chart(fig)


with col1:
    # Create a box plot using Plotly
    fig = go.Figure()

    # Add box traces for each workout frequency
    for workout_freq in filtered_df['Workout_Frequency (days/week)'].unique():
        fig.add_trace(
            go.Box(
                y=filtered_df[filtered_df['Workout_Frequency (days/week)'] == workout_freq]['Calories_Burned'],
                name=str(workout_freq),  # Set x-axis category
                #boxpoints='all',  # Display all points
                #jitter=0.3,  # Add some jitter to the points for better visibility
                #pointpos=-1.8,  # Position of points in relation to box
                marker=dict(color='#FF6692')  # Set color
            )
        )

        # Add text annotation with r value
    fig.add_annotation( 
        x=('5.5'),  # Position on the plot
        y=('1800'),  # Adjust this to position the text better
        text=f'r = 0.6',  # Display r value
        showarrow=False,
        font=dict(size=18, color='red'),
        align='right'
    )

    # Update layout
    fig.update_layout(
        title="Calories Burned by Workout Frequency",
        xaxis_title="Workout Frequency (days/week)",
        yaxis_title="Calories Burned",
        showlegend=False
    )

    # Display the Plotly figure in Streamlit
    st.plotly_chart(fig)

with col2:
    # Create a box plot using Plotly
    fig = go.Figure()

    # Add box traces for each workout frequency
    for workout_expr in filtered_df['Experience_Level'].unique():
        fig.add_trace(
            go.Box(
                y=filtered_df[filtered_df['Experience_Level'] == workout_expr]['Calories_Burned'],
                name=str(workout_expr),  # Set x-axis category
                #boxpoints='all',  # Display all points
                #jitter=0.3,  # Add some jitter to the points for better visibility
                #pointpos=-1.8,  # Position of points in relation to box
                marker=dict(color='#FF6692')  # Set color
            )
        )

    # Add text annotation with r value
    fig.add_annotation( 
        x=('3.5'),  # Position on the plot
        y=('1800'),  # Adjust this to position the text better
        text=f'r = 0.7',  # Display r value
        showarrow=False,
        font=dict(size=18, color='red'),
        align='right'
    )

    # Update layout
    fig.update_layout(
        title="Calories Burned by Workout Experience Level",
        xaxis_title="Workout Experience Level",
        yaxis_title="Calories Burned",
        showlegend=False
    )

    # Display the Plotly figure in Streamlit
    st.plotly_chart(fig)

st.header("Positive correlations analysis and conclusions:")
st.write("""
         
         - Workout attributes that are correlated are higher frequency, duration, experience and calories burned

         - **KEY FINDING**: Both Workout Frequency and Experience Level are associated with more calories burned 
         possibly driven by greater Session Duration 
    
         """)

st.title("Negative Correlations and Trends Between Category Measures")
st.write(":blue[r values displayed are Pearson correlation coefficients]")

col1, col2, col3 = st.columns([1, 1, 1], gap="large")

with col1:
    # data
    calories_burned = filtered_df['Calories_Burned']
    body_fat_percentage = filtered_df['Fat_Percentage']

    # Fit a linear regression line
    slope, intercept = np.polyfit(calories_burned, body_fat_percentage, 1)
    trend_line = slope * calories_burned + intercept

    # Create a scatter plot
    fig = go.Figure()

    # Add scatter trace for data points
    fig.add_trace(
        go.Scatter(
            x=calories_burned,
            y=body_fat_percentage,
            mode='markers',
            name='Data Points',
            marker=dict(size=5, color='#19D3F3')
        )
    )

    # Add line trace for trend line
    fig.add_trace(
        go.Scatter(
            x=calories_burned,
            y=trend_line,
            mode='lines',
            name='Trend Line',
            line=dict(color='white', dash='dash')
        )
    )

    # Add text annotation with r value
    fig.add_annotation( 
        x=max(calories_burned),  # Position on the plot
        y=max(body_fat_percentage),  # Adjust this to position the text better
        text=f'r = -0.6',  # Display r value
        showarrow=False,
        font=dict(size=18, color='#2196F3'),
        align='right'
    )

    # Update layout
    fig.update_layout(
        title='Body Fat % by Calories Burned',
        xaxis_title='Calories Burned',
        yaxis_title='Body Fat %',
        showlegend=False
    )

    # Display the Plotly figure in Streamlit
    st.plotly_chart(fig)

with col2:
    # data
    sess_dur = filtered_df['Session_Duration (hours)']
    body_fat_percentage = filtered_df['Fat_Percentage']

    # Fit a linear regression line
    slope, intercept = np.polyfit(sess_dur, body_fat_percentage, 1)
    trend_line = slope * sess_dur + intercept

    # Create a scatter plot
    fig = go.Figure()

    # Add scatter trace for data points
    fig.add_trace(
        go.Scatter(
            x=sess_dur,
            y=body_fat_percentage,
            mode='markers',
            name='Data Points',
            marker=dict(size=5, color='#19D3F3')
        )
    )

    # Add line trace for trend line
    fig.add_trace(
        go.Scatter(
            x=sess_dur,
            y=trend_line,
            mode='lines',
            name='Trend Line',
            line=dict(color='white', dash='dash')
        )
    )

    # Add text annotation with r value
    fig.add_annotation( 
        x=max(sess_dur),  # Position on the plot
        y=max(body_fat_percentage),  # Adjust this to position the text better
        text=f'r = -0.6',  # Display r value
        showarrow=False,
        font=dict(size=18, color='#2196F3'),
        align='right'
    )

    # Update layout
    fig.update_layout(
        title='Body Fat % by Session Duration',
        xaxis_title='Session Duration (hr)',
        yaxis_title='Body Fat %',
        showlegend=False
    )

    # Display the Plotly figure in Streamlit
    st.plotly_chart(fig)

with col3:
    # data
    fat_data = filtered_df['Fat_Percentage']
    weight_data = filtered_df['Weight (lbs)']

    # Fit a linear regression line
    slope, intercept = np.polyfit(weight_data, fat_data, 1)
    trend_line = slope * weight_data + intercept

    # Create a scatter plot
    fig = go.Figure()

    # Add scatter trace for data points
    fig.add_trace(
        go.Scatter(
            x=weight_data,
            y=fat_data,
            mode='markers',
            name='Data Points',
            marker=dict(size=5, color='#19D3F3')
        )
    )

    # Add line trace for trend line
    fig.add_trace(
        go.Scatter(
            x=weight_data,
            y=trend_line,
            mode='lines',
            name='Trend Line',
            line=dict(color='white', dash='dash')
        )
    )

    # Add text annotation with r value
    fig.add_annotation( 
        x=max(weight_data),  # Position on the plot
        y=max(fat_data),  # Adjust this to position the text better
        text=f'r = -0.2',  # Display r value
        showarrow=False,
        font=dict(size=18, color='#2196F3'),
        align='right'
    )

    # Update layout
    fig.update_layout(
        title='Body Fat % by Weight',
        xaxis_title='Weight (lbs)',
        yaxis_title='Body Fat %',
        showlegend=False
    )

    # Display the Plotly figure in Streamlit
    st.plotly_chart(fig)

with col1:
    # Create a box plot using Plotly
    fig = go.Figure()

    # Add box traces for each workout frequency
    for workout_freq in filtered_df['Workout_Frequency (days/week)'].unique():
        fig.add_trace(
            go.Box(
                y=filtered_df[filtered_df['Workout_Frequency (days/week)'] == workout_freq]['Fat_Percentage'],
                name=str(workout_freq),  # Set x-axis category
                marker=dict(color='#19D3F3')  # Set color
            )
        )

    # Add text annotation with r value
    fig.add_annotation( 
        x='5.3',  # Example categorical x value, replace with actual category if different
        y=max(filtered_df['Fat_Percentage']),
        text=f'r = -0.5',  # Display r value
        showarrow=False,
        font=dict(size=18, color='#2196F3'),
        align='right'
    )  

    # Update layout
    fig.update_layout(
        title="Body Fat % by Workout Frequency",
        xaxis_title="Workout Frequency (days/week)",
        yaxis_title="Body Fat %",
        showlegend=False
    )

    # Display the Plotly figure in Streamlit
    st.plotly_chart(fig)

with col2:
    # Create a box plot using Plotly
    fig = go.Figure()

    # Add box traces for each workout frequency
    for workout_expr in filtered_df['Experience_Level'].unique():
        fig.add_trace(
            go.Box(
                y=filtered_df[filtered_df['Experience_Level'] == workout_expr]['Fat_Percentage'],
                name=str(workout_expr),  # Set x-axis category
                #boxpoints='all',  # Display all points
                #jitter=0.3,  # Add some jitter to the points for better visibility
                #pointpos=-1.8,  # Position of points in relation to box
                marker=dict(color='#19D3F3')  # Set color
            )
        )

    # Add text annotation with r value
    fig.add_annotation( 
        x='3.3',  # Example categorical x value, replace with actual category if different
        y=max(filtered_df['Fat_Percentage']),
        text=f'r = -0.7',  # Display r value
        showarrow=False,
        font=dict(size=18, color='#2196F3'),
        align='right'
    )  

    # Update layout
    fig.update_layout(
        title="Body Fat % by Workout Experience Level",
        xaxis_title="Workout Experience Level",
        yaxis_title="Body Fat %",
        showlegend=False
    )

    # Display the Plotly figure in Streamlit
    st.plotly_chart(fig)

st.header("Negative correlations analysis and conclusions:")
st.write("""
         
         - All negatively correlated workout attributes are related to body fat percentage

         - There is a large transition in the data in which once calories burned is greater than 1,000 or
         session duration is more than 1.5, body fat is less than 20%

         - The large step in below 20% body fat is also observed at the highest measure with both workout frequency 
         and experience level

         - Of note, fat % and weight were only found to be weakly correlated 

         - **KEY FINDING**: Below 20% body fat is exclusive to the highest levels of certain workout measures
          and not directly to weight
    
         """)
