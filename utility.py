import os
import csv
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from scipy.spatial import distance


def load_data():
	
    # Year : 0
    # Latitude : 9
    # Longitude : 10
    # Magnitude : 12

	with open('earthquakes.csv', 'r', encoding='utf8') as f:

		reader = csv.reader(f)
		
		# skip first row
		next(reader) 

		# list of (year, longitude, latitude, magnitude)
		data = []
		
		# list of (longitude, latitude)
		points = []

		for row in reader:
			try:

				year = float(row[0])
				lon = float(row[10])
				lat = float(row[9])
				if row[12] == '':
					mag = None
				else:
					mag = float(row[12])
				
				data.append([year, lon, lat, mag])
				points.append([lon, lat])

			except Exception as e:
				pass
				#print(e)

		return data, points	
	

def lonlat_to_cartesian(points):
	# Define the radius of the sphere
	radius = 1

	# Convert the (longitude, latitude) points to spherical coordinates
	spherical_points = []
	for lon, lat in points:
		theta = np.pi/2 - np.radians(lat) # polar angle
		phi = np.radians(lon) # azimuth angle
		spherical_points.append((radius, theta, phi))

	# Convert the spherical coordinates to Cartesian coordinates
	cartesian_points = []
	for r, theta, phi in spherical_points:
		x = r * np.sin(theta) * np.cos(phi)
		y = r * np.sin(theta) * np.sin(phi)
		z = r * np.cos(theta)
		cartesian_points.append((x, y, z))
	   
	return cartesian_points


def cartesian_to_lonlat(x, y, z):
    r = math.sqrt(x**2 + y**2 + z**2)
    theta = math.acos(z/r)
    phi = math.atan2(y, x)
    theta_deg = math.degrees(theta)
    phi_deg = math.degrees(phi)
    latitude = 90 - theta_deg
    longitude = phi_deg if phi_deg >= 0 else 360 + phi_deg
    return longitude, latitude


def plot_dendogram(Z):
	dendrogram(Z)
	plt.title('Dendrogram')
	plt.xlabel('Data Point')
	plt.ylabel('Distance')
	plt.show()


# def assign_point_to_cluster(point, points, cluster_assignments):
#     cluster_points = [points[i] for i in range(len(points)) if cluster_assignments[i] == cluster_assignments[0]]
#     closest_cluster_idx = 0
#     closest_cluster_distance = distance.euclidean(point, np.mean(cluster_points, axis=0))
#     for i in range(1, len(set(cluster_assignments))):
#         cluster_points = [points[j] for j in range(len(points)) if cluster_assignments[j] == i+1]
#         cluster_distance = distance.euclidean(point, np.mean(cluster_points, axis=0))
#         if cluster_distance < closest_cluster_distance:
#             closest_cluster_idx = i
#             closest_cluster_distance = cluster_distance
#     return closest_cluster_idx

def assign_point_to_cluster(point, points, cluster_assignments, threshold=.3):
    cluster_points = [points[i] for i in range(len(points)) if cluster_assignments[i] == cluster_assignments[0]]
    closest_cluster_idx = 0
    closest_cluster_distance = distance.euclidean(point, np.mean(cluster_points, axis=0))
    for i in range(1, len(set(cluster_assignments))):
        cluster_points = [points[j] for j in range(len(points)) if cluster_assignments[j] == i+1]
        cluster_distance = distance.euclidean(point, np.mean(cluster_points, axis=0))
        if cluster_distance < closest_cluster_distance:
            closest_cluster_idx = i
            closest_cluster_distance = cluster_distance
	
    print(f"cluster distance: {closest_cluster_distance}")
    if closest_cluster_distance > threshold:
        return None
    else:
        return closest_cluster_idx


def save_cluster_data(cluster_assignments, data):
	cluster_points = {}

	for i, c in enumerate(cluster_assignments):
		
		c = str(c)
		
		if c not in cluster_points:
			cluster_points[c] = []
		
		cluster_points[c].append(data[i])


	with open('cluster_points.json', 'w') as f:
		json.dump(cluster_points, f, indent=4)
	
	return cluster_points
	
	# np.savetxt('cluster_assignments.txt', cluster_assignments, fmt='%d')
	# np.savetxt('points.txt', data, fmt='%d')


def plot_cluster_data(cluster_assignments, cartesian_points, target_cluster=None):
    cartesian_points = np.array(cartesian_points)

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    
    if target_cluster is not None:
        # Get the indices of points in the target cluster
        mask = np.array(cluster_assignments) == target_cluster
        x = cartesian_points[:,0][mask]
        y = cartesian_points[:,1][mask]
        z = cartesian_points[:,2][mask]
        
        # Scatter plot the points in the target cluster with a single color
        ax.scatter(x, y, z, c='red')
	
        # Scatter plot the non-target points with a different color
        non_target_mask = np.array(cluster_assignments) != target_cluster
        ax.scatter(cartesian_points[:,0][non_target_mask], cartesian_points[:,1][non_target_mask], cartesian_points[:,2][non_target_mask], c="gray")
    else:
        # Scatter plot all points with colors based on cluster assignments
        ax.scatter(cartesian_points[:,0], cartesian_points[:,1], cartesian_points[:,2], c=cluster_assignments, cmap='rainbow')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()
    return fig


def plot_trends(cluster):
	# Load the CSV file into a pandas dataframe
	df = pd.read_csv(f'reg-clusters\\reg_cluster{cluster}.csv')

	# Set the Year column as the index of the dataframe
	df.set_index('Year', inplace=True)

	plt.clf()

	fig = plt.figure(figsize=(10,10))

	# Create a line chart for the Count column
	plt.plot(df['Count'], label='Count')

	plt.plot(df['Average Magnitude'], label='Average Magnitude')
	# Create a line chart for the Average Magnitude column
	# Add a legend to the chart
	plt.legend()

	# Set the chart title
	plt.title('Earthquake count/magnitude per 10 year period')

	# Set the x-axis label
	plt.xlabel('Year')

	# Set the y-axis label
	plt.ylabel('Count/Average Magnitude')

	# Show the chart
	plt.show()
	return fig


def generate_reg_clusters():
	for j in range(1, 18):
		df = pd.read_csv(f"clusters\\cluster{j}.csv")
		df['Average Magnitude'].fillna(0, inplace= True)

		eq_cnt = []
		eq_avg = []
		year = []
		for i in range(1900, 2010, 10):#(int(df.iloc[0]['Year']), int(df.iloc[-1]['Year']), 10):
			filterIdx = np.where((df['Year'] >  i-10) & (df['Year'] <= i ))
			if filterIdx[0].size == 0:
				eq_avg.append(0)
				eq_cnt.append(0)
				year.append(i) 
				continue
			eq_avg.append(df.loc[filterIdx]['Average Magnitude'].mean())
			eq_cnt.append(df.loc[filterIdx]['Count'].sum())
			year.append(i)
		dataset = {'Year': year, 'Count': eq_cnt, 'Average Magnitude': eq_avg}
		df = pd.DataFrame(dataset)
		df.to_csv(f'reg-clusters\\reg_cluster{j}.csv', index= False)

# def plot_cluster_data(cluster_assignments, cartesian_points):
# 	cartesian_points = np.array(cartesian_points)
# 	#outlier_points = np.array(outlier_points)

# 	fig = plt.figure(figsize=(10,10))
# 	ax = fig.add_subplot(111, projection='3d')
# 	ax.scatter(cartesian_points[:,0], cartesian_points[:,1], cartesian_points[:,2], c=cluster_assignments, cmap='rainbow')
# 	#ax.scatter(outlier_points[:,0], outlier_points[:,1], outlier_points[:,2], c="blue")
# 	ax.set_xlabel('X')
# 	ax.set_ylabel('Y')
# 	ax.set_zlabel('Z')
# 	plt.show()
# 	return fig


# def plot_single_cluster(cluster_assignments, cartesian_points, cluster_num):
# 	# Get the points in cluster 1
#     cluster_points = [cartesian_points[i] for i in range(len(cartesian_points)) if cluster_assignments[i] == cluster_num]
#     x, y, z = zip(*cluster_points)

#     # Create a 3D scatter plot of the points in cluster 1
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.scatter(x, y, z)
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
#     plt.show()
#     return fig


def save_points_csv(cluster_points, cluster_name):

	# find earliest and latest year in cluster

	year_points = {}

	#years = []
	for point in cluster_points[cluster_name]:
		if point[0] not in year_points:
			year_points[point[0]] = {}
			year_points[point[0]]["count"] = 1
			year_points[point[0]]["magnitude"] = []
			if point[3]:
				year_points[point[0]]["magnitude"].append(point[3])
		else:
			year_points[point[0]]["count"] += 1
			if point[3]:
				year_points[point[0]]["magnitude"].append(point[3])
		#years.append(point[0])

	# with open('year_points.json', 'w') as f:
	# 	json.dump(year_points, f, indent=4)

	#for i in range(min(years), max(years)+1):

	with open(f'clusters\\cluster{cluster_name}.csv', 'w', newline='') as csvfile:
		writer = csv.writer(csvfile)
		writer.writerow(["Year", "Count", "Average Magnitude"])
		for year in year_points.keys():
			if len(year_points[year]["magnitude"]) > 0:
				average_magnitude = sum(year_points[year]["magnitude"]) / len(year_points[year]["magnitude"])
			else:
				average_magnitude = None
			writer.writerow([year, year_points[year]["count"], average_magnitude])
			

# def partition_data():

# 	df = pd.read_csv('earthquakes.csv')

# 	# remove any points that are missing these features
# 	df = df.dropna(subset=['Year'])
# 	df = df.dropna(subset=['Longitude'])
# 	df = df.dropna(subset=['Latitude'])
# 	df = df.dropna(subset=['Mag'])

# 	train_df, test_df = train_test_split(df, test_size=0.2)

# 	train_df.to_csv('train_data.csv', index=False)
# 	test_df.to_csv('test_data.csv', index=False)


# def separate_outliers(cartesian_points):
# 	# Calculate the z-scores for each coordinate dimension
# 	z_scores = np.abs((cartesian_points - np.mean(cartesian_points, axis=0)) / np.std(cartesian_points, axis=0))

# 	# Identify outliers using a threshold z-score value
# 	outlier_threshold = 4
# 	outliers = np.where(np.any(z_scores > outlier_threshold, axis=1))[0]

# 	# Remove outliers and store them in a separate list
# 	outlier_points = [cartesian_points[i] for i in outliers]
# 	print(outlier_points)
# 	cartesian_points = [cartesian_points[i] for i in range(len(cartesian_points)) if i not in outliers]
	
# 	return cartesian_points, outlier_points