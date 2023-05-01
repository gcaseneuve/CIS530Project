from scipy.cluster.hierarchy import linkage, fcluster
from utility import *
from PIL import Image

import streamlit as st



def main():

	data, points = load_data()

	# convert (longitude, latitude) to (x, y, z)
	cartesian_points = lonlat_to_cartesian(points)

	#cartesian_points, outlier_points = separate_outliers(cartesian_points)

	# hierarchical clustering
	Z = linkage(cartesian_points, method='ward')
	#plot_dendogram(Z)
	distance_threshold = 5
	cluster_assignments = fcluster(Z, distance_threshold, criterion='distance')


	# l = lonlat_to_cartesian([(-74,40)])
	# p = l[0]#(5,20,-50)
	# print(p)

	# print(assign_point_to_cluster(p, cartesian_points, cluster_assignments))

	# print(cartesian_to_lonlat(p[0], p[1], p[2]))

	#generate_reg_clusters()

	cluster_points = save_cluster_data(cluster_assignments, data)
	# for i in range(1, 18):
	# 	save_points_csv(cluster_points, str(i))

	fig = plot_cluster_data(cluster_assignments, cartesian_points)

	st.title("Predicting Earthquakes by Region")

	lat = st.number_input('Insert a latitude')
	lon = st.number_input('Insert a longitude')
	

	l = lonlat_to_cartesian([(lon, lat)])
	p = l[0]
	print(p)
	#st.write('converted to (x,y,z): ', p)
	nearest_cluster = assign_point_to_cluster(p, cartesian_points, cluster_assignments)
	st.write('Nearest Region: ', nearest_cluster)

	# SHOW HARD-CODED GREECE EXAMPLE
	predicted_mag = 3.474951
	if nearest_cluster == 5:
		st.write("Predicted Magnitude: ", predicted_mag)
	
	if nearest_cluster:
		fig = plot_cluster_data(cluster_assignments, cartesian_points, nearest_cluster)
	else:
		fig = plot_cluster_data(cluster_assignments, cartesian_points)
	st.pyplot(fig)

	if nearest_cluster:
		fig2 = plot_trends(nearest_cluster)
		st.pyplot(fig2)

	# GREECE EXAMPLE
	if nearest_cluster == 5:
		image = Image.open('image.png')

		st.image(image, width=800)


if __name__ == '__main__':
	main()
