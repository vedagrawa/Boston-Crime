"""
Shreesh Dassarkar, Ved Agrawal, Alexander Zhen, and Ayush Ghose

DS2500

12/4/2023

Final Project: Analyzing Crime data across the City of Boston

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point, Polygon
import seaborn as sns
from sklearn.linear_model import LinearRegression


# List of file paths
crime_files = [
    "crime_incidents2015.csv",
    "crime_incidents2016.csv",
    "crime_incidents2017.csv",
    "crime_incidents2018.csv",
    "crime_incidents2019.csv",
    "crime_incidents2020.csv",
    "crime_incidents2021.csv",
    "crime_incidents2022.csv"
]

na_values = ['', ' ', 'NA', 'N/A', 'NaN']

dtype_mapping = {
    'id': str,
    'Incident_Number': int,
    'Offense_Code': int,
    'Offense_Code_Group': str,
    'Offense_Description': str,
    'District': str,
    'Reporting_Area': int,
    'Shooting': str,
    'Occurred_on_Date': str,
    'Year': int,
    'Month': int,
    'Day_of_Week': str,
    'Hour': int,
    'UCR_Part': str,
    'Street': str,
    'Lat': float,
    'Long': float,
    'Location': str
}

CRS = {'init': 'epsg:4326'}

CITY_SHP = "City_of_Boston_Boundary.shp"

schools = [
    {"name": "Northeastern", "coords": (-71.08986774576756, 42.34004671721457)},
    {"name": "BU", "coords": (-71.09955963851051, 42.34972483774112,)},
    {"name": "Suffolk", "coords": (-71.06112177331377, 42.35793902130814,)},
    {"name": "Emmanuel", "coords": (-71.10216091593881, 42.34125472008568, )},
    {"name": "UMass Boston", "coords": (-71.04195661917427, 42.314383379259944)},
    {"name": "BC", "coords": (-71.16586916005554, 42.337293270935476, )},
    {"name": "Roxbury CC", "coords": (-71.0953466807458, 42.32958193786075)}]

colors = ["red", "chocolate", "violet", "seagreen", "deeppink", "gold"
          , "orange"]

CRIME_COL = "OFFENSE_CODE_GROUP"

SHOOTING_COL = "SHOOTING"

DATE_COL = "OCCURRED_ON_DATE"

def read_and_concatenate_files(file_paths):
    """
    Reads and concatenates multiple CSV files into a single DataFrame.

    Args:
    file_paths (list of str): List of file paths to the CSV files.

    Returns:
    pandas.DataFrame: Concatenated DataFrame containing all data from the input files.
    """
    
    na_values = ['', ' ', 'NA', 'N/A', 'NaN']
    
    dfs = [pd.read_csv(file_path, na_values=na_values, low_memory=False) for file_path in file_paths]
    
    return pd.concat(dfs, ignore_index=True)

def find_most_frequent_crime(crime_data):
    """
    Identifies the most frequent crime type from the crime data.

    Args:
    crime_data (pandas.DataFrame): DataFrame containing crime data.

    Returns:
    tuple: A tuple containing the most frequent crime type and its count.
    """
    
    crime_data = crime_data[crime_data['OFFENSE_CODE_GROUP'] != 'Other']
    
    most_frequent_crime = crime_data['OFFENSE_CODE_GROUP'].value_counts().idxmax()
    
    most_frequent_crime_count = crime_data['OFFENSE_CODE_GROUP'].value_counts().max()
    
    return most_frequent_crime, most_frequent_crime_count

def analyze_crime_trends(crime_data):
    """
    Analyzes crime trends over years and identifies top increasing and decreasing crimes.

    Args:
    crime_data (pandas.DataFrame): DataFrame containing crime data.

    Returns:
    tuple: Two pandas.Series for top 5 increasing and decreasing crimes respectively.
    """
    
    crime_data = crime_data[crime_data['OFFENSE_CODE_GROUP'] != 'Other']
    
    crime_trends = crime_data.groupby(['YEAR', 'OFFENSE_CODE_GROUP']).size().unstack(fill_value=0)
    
    crime_trends_percentage_change = crime_trends.pct_change().dropna()
    
    overall_trend = crime_trends_percentage_change.sum()
    
    top_5_increasing_crimes = overall_trend.nlargest(5)
    
    top_5_decreasing_crimes = overall_trend.nsmallest(5)
    
    return top_5_increasing_crimes, top_5_decreasing_crimes

def find_areas_with_most_crime(crime_data):
    """
    Identifies the top 5 districts with the most crime incidents.

    Args:
    crime_data (pandas.DataFrame): DataFrame containing crime data.

    Returns:
    pandas.Series: Series containing the top 5 districts with most crimes.
    """
    
    return crime_data['DISTRICT'].value_counts().head(5)

def find_top_crimes_in_districts(crime_data, districts, excluded_crimes=['Motor Vehicle Accident Response', 'Medical Assistance', 'Other']):
    """
    Finds the top 3 crimes for each specified district, excluding certain types of crimes.

    Args:
    
        crime_data (pandas.DataFrame): DataFrame containing crime data.
    
    districts (list of str): List of districts to analyze.
    
    excluded_crimes (list of str): List of crime types to exclude from the analysis.

    Returns:
    dict: Dictionary where keys are districts and values are pandas.Series of top 3 crimes.
    """
    
    top_crimes = {}
    
    for district in districts:
        district_data = crime_data[crime_data['DISTRICT'] == district]
        
        for crime in excluded_crimes:
            district_data = district_data[district_data['OFFENSE_CODE_GROUP'] != crime]
        
        top_crimes_in_district = district_data['OFFENSE_CODE_GROUP'].value_counts().head(3)
        
        top_crimes[district] = top_crimes_in_district
    
    return top_crimes

def find_top_crimes_excluding_specific(crime_data, districts, excluded_crimes):
    """
    Finds the top 3 crimes in specified districts, excluding certain crimes based on description.

    Args:
    
        crime_data (pandas.DataFrame): DataFrame containing crime data.
    
    districts (list of str): List of districts to analyze.
    
    excluded_crimes (list of str): List of crime descriptions to exclude.

    Returns:
    dict: Dictionary where keys are districts and values are pandas.Series of top 3 crimes based on descriptions.
    """
    
    top_crimes = {}
    
    for district in districts:
        district_data = crime_data[crime_data['DISTRICT'] == district]
        
        district_data = district_data[~district_data['OFFENSE_DESCRIPTION'].isin(excluded_crimes)]
        
        top_crimes_in_district = district_data['OFFENSE_DESCRIPTION'].value_counts().head(3)
        
        top_crimes[district] = top_crimes_in_district
    
    return top_crimes

def plot_top_crimes_trend_with_connected_line(crime_data, district, top_crimes):
    """
    Plots the trend of top crimes in a district with connected lines showing actual data points and trend line.

    Args:
    
        crime_data (pandas.DataFrame): DataFrame containing crime data.
    
    district (str): The district to analyze.
    
    top_crimes (list of str): List of top crimes to plot.

    Returns:
    None: This function plots a graph and does not return anything.
    """
    
    plt.figure(figsize=(12, 8))
    
    full_years = np.arange(2015, 2023)

    for crime in top_crimes:
        
        crime_data_specific = crime_data[(crime_data['DISTRICT'] == district) & (crime_data['OFFENSE_DESCRIPTION'] == crime)]
        
        yearly_counts = crime_data_specific.groupby('YEAR').size().reindex(full_years, fill_value=0)
        
        X = yearly_counts.index.values.reshape(-1, 1)
        
        y = yearly_counts.values

        model = LinearRegression()
        
        model.fit(X, y)
        
        predicted = model.predict(X)

        plt.plot(X.flatten(), y, marker='o', label=f'Actual {crime}')
        
        plt.plot(X.flatten(), predicted, linestyle='--', label=f'Trend {crime}')

    plt.title(f'Top Crimes Trend in District {district} (Based on Description)')
    
    plt.xlabel('Year')
    
    plt.xticks(X.flatten(), X.flatten(), rotation=45)
    
    plt.ylabel('Crime Count')
    
    plt.legend()
    
    plt.grid(True)
    
    plt.show()
    
def forecast_crime(crime_data, forecast_years=5):
    """
    Forecasts future crime counts based on historical data using linear regression.

    Args:
    
        crime_data (pandas.DataFrame): DataFrame containing crime data with a 'YEAR' column.
    
        forecast_years (int, optional): Number of years into the future to forecast. Defaults to 5.

    Returns:
    tuple: 
        
        - X (numpy.ndarray): Array of years used for model training.
        
        - y (numpy.ndarray): Array of crime counts corresponding to each year in X.
        
        - future_years (numpy.ndarray): Array of future years for which crime count is forecasted.
        
        - predicted_counts (numpy.ndarray): Array of predicted crime counts for future_years.
    """
    
    yearly_crime_counts = crime_data['YEAR'].value_counts().sort_index()
    
    X = yearly_crime_counts.index.values.reshape(-1, 1)
    
    y = yearly_crime_counts.values
    
    model = LinearRegression()
    
    model.fit(X, y)
    
    future_years = np.array([X[-1, 0] + i for i in range(1, forecast_years + 1)]).reshape(-1, 1)
    
    predicted_counts = model.predict(future_years).astype(int)
    
    return X, y, future_years, predicted_counts


def plot_crime_forecast(X, y, future_years, predicted_counts):
    """
    Plots the historical crime counts and forecasts future crime trends.

    Args:
    
        X (numpy.ndarray): Array of years used for model training.
    
        y (numpy.ndarray): Array of actual crime counts corresponding to each year in X.
    
        future_years (numpy.ndarray): Array of future years for which crime count is forecasted.
    
        predicted_counts (numpy.ndarray): Array of predicted crime counts for future_years.
    
    Returns:
    None: This function plots a graph and does not return any value.
    """
    
    plt.figure(figsize=(10, 6))
    
    plt.scatter(X, y, color='blue', label='Actual Crime Counts')
    
    plt.plot(X, LinearRegression().fit(X, y).predict(X), color='red', label='Fitted Line')
    
    plt.plot(future_years, predicted_counts, color='green', linestyle='--', label='Forecast')
    
    plt.xlabel('Year')
    
    plt.ylabel('Crime Count')
    
    plt.title('Crime Count Forecast')
    
    plt.legend()
    
    plt.show()
    
    
def read_shp(path, crs):
    """
    Read in the shapefile data and set up the proper CRS.

    Parameters:
    
        - path (str): Path to the shapefile.
    
        - crs (str): Coordinate Reference System.

    Returns:
    - gpd.GeoDataFrame: GeoDataFrame containing shapefile data.
    
    """
    shape = gpd.read_file(path)
    
    shape.crs = crs
    
    return shape


def make_crime_plot(city_map, df, column, identifier, schools, colors,
                    
                    coord_col, name_col, school_gdf, 
                    
                    title, map_col = "white", 
                    
                    edge_col = "black", point_col = "blue", marker = "o", 
                    
                    point_size = 1, school_size = 120,
                    
                    xlab = "Longitude", ylab = "Latitude"):
    """
    Given relevant data, plot all the data points for a certain type
    of crime on the map of Boston, plotting with it the colleges in the
    area as well.

    Parameters:
    - city_map (gpd.GeoDataFrame): GeoDataFrame of the city map.
    - df (pd.DataFrame): DataFrame containing crime data.
    - column (str): Column in df containing crime types.
    - identifier (str): Identifier for the specific crime type.
    - schools (list): List of schools.
    - colors (list): List of colors for schools.
    - coord_col (str): Column in schools containing coordinates.
    - name_col (str): Column in schools containing school names.
    - school_gdf (gpd.GeoDataFrame): GeoDataFrame containing school data.
    - title (str): Title for the plot.
    - map_col (str): Color of the city map.
    - edge_col (str): Color of the map edges.
    - point_col (str): Color of crime points.
    - marker (str): Marker style for crime points.
    - point_size (int): Size of crime points.
    - school_size (int): Size of school points.
    - xlab (str): Label for the x-axis.
    - ylab (str): Label for the y-axis.
    """
    
    # Plot the map of the city
    
    fig, ax = plt.subplots(figsize = (15, 15))
    
    city_map.plot(ax = ax, color = map_col, edgecolor = edge_col)
    
    # Plot the crime
    
    df[df[column] == identifier].plot(ax = ax, markersize = point_size, 
                                      
                                      color = point_col, marker = marker)
    
    # Plot the schools
    
    for school, color in zip(schools, colors):
        
        school_set = school_gdf[school_gdf.geometry == Point(school[coord_col])]
        
        school_set.plot(ax = ax, markersize  = school_size, 
                        
                        color = color, marker = marker, label=school[name_col])
    
    ax.legend()
    
    plt.xlabel(xlab)
    
    plt.ylabel(ylab)
    
    plt.title(title)

    plt.show()
    
def df_to_datetime(df, column):
    """
    Given a dataframe and a column that indicates the date of an event,
    return a new dataframe.

    Parameters:
    
        - df (pd.DataFrame): Input DataFrame.
   
        - column (str): Column containing date information.

    Returns:
    - pd.DataFrame: Modified DataFrame with year and month columns.
    
    """
    df[column] = pd.to_datetime(df[column])
    
    # Extract year and month
    
    df['year'] = df[column].dt.year
    
    df['month'] = df[column].dt.month
    
    # Generate modified dataframe
    
    df_2 = df.groupby(['year', 'month']).size().reset_index(name='occurences')
    
    return df_2

def make_heatmap(df, xlab, ylab, title, fmt = ".2f", cmap = "binary"):
    """
    Create a heatmap from a DataFrame.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - xlab (str): Label for the x-axis.
    - ylab (str): Label for the y-axis.
    - title (str): Title for the plot.
    - fmt (str): Format for annotating the heatmap cells.
    - cmap (str): Colormap for the heatmap.
    
    """
    data = df.pivot_table(index = 'month', columns = 'year', 
                          
                          values = 'occurences', aggfunc = 'sum')
    
    plt.figure(figsize = (10, 7))
    
    sns.heatmap(data, fmt = fmt, cmap = cmap, annot = True)
    
    plt.xlabel(xlab)
    
    plt.ylabel(ylab)
    
    plt.title(title)
    
    plt.show()


def main():
    """
    Main function to perform crime data analysis and visualization.
    Combines various functions to read crime data, analyze trends, and visualize results.
    
    """

    # Read and concatenate crime data files into a single DataFrame
    
    all_data = read_and_concatenate_files(crime_files)

    # Identify and print the most frequent crime type and its count
   
    most_frequent_crime, count = find_most_frequent_crime(all_data)
    
    print(f"Most Frequent Crime: {most_frequent_crime} (Count: {count})")

    # Analyze and print the top 5 increasing and decreasing crime trends
    
    increasing_crimes, decreasing_crimes = analyze_crime_trends(all_data)
    
    print("Top 5 Increasing Crimes:", increasing_crimes)
    
    print("Top 5 Decreasing Crimes:", decreasing_crimes)

    # Find and print the top 5 districts with the most crime incidents
    
    top_districts = find_areas_with_most_crime(all_data)
    
    print("Top 5 Districts with Most Crimes:", top_districts)

    # Determine and print the top 3 crimes in specific districts
    
    districts = ['D4', 'B2', 'C11']
    
    top_crimes_by_district = find_top_crimes_in_districts(all_data, districts)
    
    for district, crimes in top_crimes_by_district.items():
        
        print(f"Top 3 Crimes in District {district}:", crimes)

    # Identify the top crimes in specified districts while excluding certain crimes
    
    top_districts = ['B2', 'C11', 'D4']
   
    excluded_crimes = ['M/V - LEAVING SCENE - PROPERTY DAMAGE', 'LARCENY THEFT FROM BUILDING', 'SICK/INJURED/MEDICAL - PERSON']
    
    updated_top_crimes_by_district_description  = find_top_crimes_excluding_specific(all_data, top_districts, excluded_crimes)

    # Forecast crime counts for the next 5 years and plot the forecast

    X, y, future_years, predicted_counts = forecast_crime(all_data, forecast_years=5)
   
    plot_crime_forecast(X, y, future_years, predicted_counts)

    # Plot trend lines of top crimes in specified districts
    
    for district, crimes in updated_top_crimes_by_district_description.items():
        
        plot_top_crimes_trend_with_connected_line(all_data, district, crimes.index[:3])
        
    
    # For every data point in the dataframe, create a "Point" based on 
    
    # latitude and longitude coordinates
    
    geometry = [Point(xy) for xy in zip(all_data['Long'], all_data['Lat'])]
    
    # Convert the dataframe to a geodataframe
    
    geo_df = gpd.GeoDataFrame(all_data, geometry = geometry, crs = CRS)
    
    # Shapefile processing: Read in the shapefile for the map of Boston
    
    # and the one which contains the locations of all the police stations
    
    city_map = read_shp(CITY_SHP, CRS)
    
    # Make a new geodataframe with the school's data
    
    school_geometry = [Point(xy) for xy in [school["coords"] for 
                                            school in schools]]
    
    school_gdf = gpd.GeoDataFrame(geometry = school_geometry, crs = CRS)
    
    # Plot 1: Larceny in Boston
    
    larceny_plot = make_crime_plot(city_map, geo_df, CRIME_COL, "Larceny", 
                                    
                                   schools, colors, "coords", "name", 
                                    
                                   school_gdf, "Larceny")
    
    # Plot 2: Simple Assault in Boston
    
    assault_plot = make_crime_plot(city_map, geo_df, CRIME_COL, "Simple Assault", 
                                    
                                   schools, colors, "coords", "name", 
                                    
                                    school_gdf, "Simple Assault")
    
    # Plot 3: Shootings in Boston
    
    shooting_plot = make_crime_plot(city_map, geo_df, SHOOTING_COL, "Y", 
                                    
                                    schools, colors, "coords", "name", 
                                    
                                    school_gdf, "Shootings")
    
    # Generate a new df so that we can see crimes in terms of years and months
    
    datetime_df = df_to_datetime(all_data, DATE_COL)
    
    # Plot all the data points by year and month
    
    heatmap = make_heatmap(datetime_df, "Years", "Months (in numerical format)",
                      "Crime in Boston (2015 to 2022)")

        
if __name__ == "__main__":
    main()
