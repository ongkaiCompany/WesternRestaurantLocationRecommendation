# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 14:50:47 2024
For Submission
@author: ryank, chenghow, wengkai, zeelin
"""

import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pickle
import folium
import plotly.express as px
import plotly.graph_objects as go
#need to download matplotlib
from folium.plugins import MarkerCluster
import os
from sklearn.cluster import KMeans, Birch, AffinityPropagation, AgglomerativeClustering
from sklearn.cluster import DBSCAN

if "dataframe" not in st.session_state:
    st.session_state.dataframe = None
    
if "X" not in st.session_state:
    st.session_state.X = None

if "X_scaled" not in st.session_state:
    st.session_state.X_scaled = None

def main():
    #st.write(path_without_drive)
    st.sidebar.markdown("## Select the analysis step")
    step = st.sidebar.radio("Please select an area that you would like to explore", ["Introduction","Upload Dataset", "Visualise Clusters", "Display Rating Plots", "Select Suburb or Road"])
    
    if step == "Introduction":
        st.write("<h1 style='width: 800px;'>Western Specialty Restaurant Location Recommendation in Singapore</h1>", unsafe_allow_html=True)
        st.markdown("* This interative dashboard uses unsupervised machine learning to recommend optimal locations for establishing a new Western restaurant in Singapore's competitive culinary scene.")
        st.markdown("* The recommendation takes into account areas with a demand for Western cuisine, while also considering low ratings of existing Western restaurants in those areas.")
        st.markdown("* Please select the options on the left to explore further :smiley:")
        st.image("https://www.marinabaysands.com/content/dam/marinabaysands/guides/foodie-guide/western-restaurants/masthead.jpg", caption="Western Restaurant", use_column_width=True)
        
    elif step == "Upload Dataset":
    # Title of the web app
        st.write("<h1 style='width: 800px;'>Upload Dataset of Western Restaurants in Singapore</h1>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
        if uploaded_file is not None:
            # Read the uploaded CSV file as a pandas DataFrame
            df = pd.read_csv(uploaded_file)
        
            # Store the DataFrame in session state
            st.session_state.dataframe = df
        
            # Display a success message
            st.success("CSV file uploaded successfully!")
            
            X = df[['latitude', 'longitude', 'rating']]
            
            scaler = MinMaxScaler()
            X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
        
            # Store X and X_scaled in session state
            st.session_state.X = X
            st.session_state.X_scaled = X_scaled

            # Display the uploaded DataFrame if it exists in session state
        if st.session_state.dataframe is not None:
            st.write("✅ **Uploaded DataFrame:**")
            st.write(st.session_state.dataframe)
            st.write("There are",len(st.session_state.dataframe),"records.")
        
        if st.session_state.X is not None:
            st.write("✅ **Feature Selection:**")
            st.write(st.session_state.X)

        # Display X_scaled if it exists in session state
        if st.session_state.X_scaled is not None:
            st.write("✅ **Feature Scaling using MinMaxScaler:**")
            st.write(st.session_state.X_scaled)
            
            
    
    elif step == 'Visualise Clusters':
        st.write("<h1 style='width: 800px;'>Visualisation of Clusters</h1>", unsafe_allow_html=True)
          
        
        message = "Nothing to display here. Please upload a dataset. Thank you."
        
        
        
        ###############################################################################################
        
        # 1. K Means Clustering
        
        st.subheader("1. K Means Clustering")
        with st.container():
           st.write("✅ **The most optimal hyperparameter values:**")
           st.write("- n_clusters: 5")
           st.write("- init: k-means++")
           st.write("- max_iter: 300")
           st.write("- tol: 0.0001")
           st.write("✅ **Toggle the widgets below to modify the Hyperparameter Values**")
           col1, col2 = st.columns(2)

           # Slider for the number of clusters
           with col1:
              n_clusters = st.slider("Number of Clusters", min_value=2, max_value=5, value=5, step=1)

           # Radio button for initialization method
           with col2:
              init_method = st.radio("Initialization Method", options=['k-means++', 'random'], index=0)

           # Slider for maximum number of iterations
           with col1:
              max_iter = st.slider("Maximum Number of Iterations", min_value=100, max_value=1000, value=300, step=100)

           # Slider for tolerance
           with col2:
              tolerance = st.slider("Tolerance", min_value=0.0001, max_value=0.01, value=0.0001, step=0.0001)

        
           
        
        
        kmean_df = st.session_state.dataframe
        if kmean_df is not None:            
            
           #bestKM = KMeans(n_clusters=5, random_state=42, n_init = 'auto', init = 'k-means++', max_iter = 300, tol = 0.0001)
           bestKM = KMeans(n_clusters=n_clusters, random_state=42, n_init = 'auto', init = init_method, max_iter = max_iter, tol = tolerance)
           bestKM.fit(st.session_state.X_scaled)
           kmean_df['cluster'] = bestKM.predict(st.session_state.X_scaled)          
           
           if 'name' in kmean_df.columns:
             
             def map_cluster_to_color(cluster_number):
                 cluster_colors = {
                     0: {'color': 'Cluster 0', 'label': 'Cluster 0'},
                     1: {'color': 'Cluster 1', 'label': 'Cluster 1'},
                     2: {'color': 'Cluster 2', 'label': 'Cluster 2'},
                     3: {'color': 'Cluster 3', 'label': 'Cluster 3'},
                     4: {'color': 'Cluster 4', 'label': 'Cluster 4'}
                 }
                 return cluster_colors[cluster_number]['color']
             
             kmean_df['cluster_label'] = kmean_df['cluster'].map(map_cluster_to_color)
             cluster_labels = {0: 'Cluster 2', 1: 'Cluster 0', 2: 'Cluster 1', 3: 'Cluster 3'}
             kmean_df['custom_cluster_label'] = kmean_df['cluster'].map(cluster_labels)
             
             cluster_avg_rating = kmean_df.groupby('cluster')['rating'].mean().reset_index()
             cluster_avg_rating.columns = ['cluster', 'avg_rating']
             kmean_df = pd.merge(kmean_df, cluster_avg_rating, on='cluster', how='left')


             fig = px.scatter_mapbox(kmean_df, 
                                lat="latitude", 
                                lon="longitude", 
                                color="cluster_label", 
                                hover_name="name", 
                                hover_data={"cluster": False, "rating": True, "avg_rating": True, "cluster_label": False},
                                zoom=10, 
                                mapbox_style="carto-positron",
                                title="Map for K-Means Clustering")
             
             fig.update_layout(mapbox=dict(
                 center=dict(lat=kmean_df['latitude'].mean(), lon=kmean_df['longitude'].mean()),
             ))
             
             fig.update_traces(marker=dict(size=10, opacity=0.8),
                          hovertemplate="<b>%{hovertext}</b><br>" +
                                        "Cluster: %{customdata[0]}<br>" +
                                        "Rating: %{customdata[1]}<br>" +
                                        "Cluster Average Rating: %{customdata[2]}<br>" +
                                        "<extra>Cluster Label: %{customdata[3]}</extra>")
             ########################################################################################
             
             cluster_stats = kmean_df.groupby('cluster').agg(
                 no_of_reviews=('no_of_reviews', 'sum'),
                 avg_rating=('rating', 'mean')
             ).reset_index()
             fig2 = px.bar(
                 cluster_stats, 
                 x='cluster', 
                 y='no_of_reviews',
                 title='Number of Reviews per Cluster',
                 labels={'no_of_reviews': 'Total Reviews'},
                 hover_data={'avg_rating': ':.2f'}  # Display average rating on hover, formatted to 2 decimal places
                 )    
             st.plotly_chart(fig)
             st.plotly_chart(fig2, use_container_width=True)
           else:
            st.write("The 'name' column does not exist in the DataFrame.")
        else:
          st.write(message)

        
        ###############################################################################################
        
        # 2. BIRCH Clustering
        st.subheader("2. Birch Clustering")
        with st.container():
           st.write("✅ **The most optimal hyperparameter values:**")
           st.write("- branching_factor: 25")
           st.write("- n_clusters: 4")
           st.write("- threshold: 0.1")
           st.write("✅ **Toggle the widgets below to modify the Hyperparameter Values**")
           # Use columns to display widgets side by side
           col1, col2, col3 = st.columns(3)

           # Slider for branching factor
           with col1:
              branching_factor = st.slider("Branching Factor", min_value=2, max_value=100, value=25, step=1)

           # Slider for number of clusters
           with col2:
              n_clusters = st.slider("Number of Clusters", min_value=2, max_value=4, value=4, step=1)

           # Slider for threshold
           with col3:
              threshold = st.slider("Threshold", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
        birch_df = st.session_state.dataframe
        if birch_df is not None:
            #bestbirch = Birch(branching_factor = 25, n_clusters = 4, threshold = 0.1)
            bestbirch = Birch(branching_factor=branching_factor, n_clusters=n_clusters, threshold=threshold)
            bestbirch.fit(st.session_state.X_scaled)
            birch_df['cluster'] = bestbirch.predict(st.session_state.X_scaled)
            
            

            if 'name' in birch_df.columns:              
              def map_cluster_to_color(cluster_number):
                  cluster_colors = {
                      0: {'color': 'Cluster 0', 'label': 'Cluster 0'},
                      1: {'color': 'Cluster 1', 'label': 'Cluster 1'},
                      2: {'color': 'Cluster 2', 'label': 'Cluster 2'},
                      3: {'color': 'Cluster 3', 'label': 'Cluster 3'}
                  }
                  return cluster_colors[cluster_number]['color']
              
              birch_df['cluster_label'] = birch_df['cluster'].map(map_cluster_to_color)
              cluster_labels = {0: 'Cluster 2', 1: 'Cluster 0', 2: 'Cluster 1', 3: 'Cluster 3'}
              birch_df['custom_cluster_label'] = birch_df['cluster'].map(cluster_labels)
              
              cluster_avg_rating = birch_df.groupby('cluster')['rating'].mean().reset_index()
              cluster_avg_rating.columns = ['cluster', 'avg_rating']
              birch_df = pd.merge(birch_df, cluster_avg_rating, on='cluster', how='left')


              fig = px.scatter_mapbox(birch_df, 
                                 lat="latitude", 
                                 lon="longitude", 
                                 color="cluster_label", 
                                 hover_name="name", 
                                 hover_data={"cluster": False, "rating": True, "avg_rating": True, "cluster_label": False},
                                 zoom=10, 
                                 mapbox_style="carto-positron",
                                 title="Map for Birch Clustering")
              
              fig.update_layout(mapbox=dict(
                  center=dict(lat=birch_df['latitude'].mean(), lon=birch_df['longitude'].mean()),
              ))
              
              fig.update_traces(marker=dict(size=10, opacity=0.8),
                           hovertemplate="<b>%{hovertext}</b><br>" +
                                         "Cluster: %{customdata[0]}<br>" +
                                         "Rating: %{customdata[1]}<br>" +
                                         "Cluster Average Rating: %{customdata[2]}<br>" +
                                         "<extra>Cluster Label: %{customdata[3]}</extra>")

              cluster_stats = birch_df.groupby('cluster').agg(
                  no_of_reviews=('no_of_reviews', 'sum'),
                  avg_rating=('rating', 'mean')
              ).reset_index()
              fig3 = px.bar(
                  cluster_stats, 
                  x='cluster', 
                  y='no_of_reviews',
                  title='Number of Reviews per Cluster',
                  labels={'no_of_reviews': 'Total Reviews'},
                  hover_data={'avg_rating': ':.2f'}  # Display average rating on hover, formatted to 2 decimal places
                  )    
              st.plotly_chart(fig)
              st.plotly_chart(fig3, use_container_width=True)
              
            else:
             st.write("The 'name' column does not exist in the DataFrame.")
        else:
           st.write(message)
        
        
        #3. AP Clustering
        st.subheader("3. Affinity Propagation Clustering")
        with st.container():
           st.write("✅ **The most optimal hyperparameter values:**")
           st.write("- Preference: -5")
           st.write("- Damping: 0.7")
           st.write("✅ **Toggle the widgets below to modify the Hyperparameter Values**")
           # Use columns to display widgets side by side
           col1, col2 = st.columns(2)

           # Slider for preference
           with col1:
              preference = st.slider("Preference", min_value=-10, max_value=-1, value=-5, step=1)

           # Slider for damping
           with col2:
              damping = st.slider("Damping", min_value=0.5, max_value=1.0, value=0.7, step=0.1)

        ap_df = st.session_state.dataframe
        if ap_df is not None:
            bestap = AffinityPropagation(affinity='euclidean', damping=damping, preference=preference, random_state = None, verbose = False, max_iter=200, copy=True, convergence_iter=15).fit(st.session_state.X_scaled)
            #bestap = AffinityPropagation(affinity='euclidean', damping=0.7, preference=-5, random_state = None, verbose = False, max_iter=200, copy=True, convergence_iter=15).fit(st.session_state.X_scaled)
            ap_df['cluster'] = bestap.predict(st.session_state.X_scaled)
            
            
            if 'name' in ap_df.columns:
              
              def map_cluster_to_color(cluster_number):
                  cluster_colors = {
                      0: {'color': 'Cluster 0', 'label': 'Cluster 0'},
                      1: {'color': 'Cluster 1', 'label': 'Cluster 1'},
                      2: {'color': 'Cluster 2', 'label': 'Cluster 2'},
                      3: {'color': 'Cluster 3', 'label': 'Cluster 3'},
                      4: {'color': 'Cluster 4', 'label': 'Cluster 4'},
                      5: {'color': 'Cluster 5', 'label': 'Cluster 5'},
                      6: {'color': 'Cluster 6', 'label': 'Cluster 6'},
                      7: {'color': 'Cluster 7', 'label': 'Cluster 7'},
                      8: {'color': 'Cluster 8', 'label': 'Cluster 8'}
                  }
                  return cluster_colors[cluster_number]['color']
              
              ap_df['cluster_label'] = ap_df['cluster'].map(map_cluster_to_color)
              cluster_labels = {0: 'Cluster 2', 1: 'Cluster 0', 2: 'Cluster 1', 3: 'Cluster 3'}
              ap_df['custom_cluster_label'] = ap_df['cluster'].map(cluster_labels)
              
              cluster_avg_rating = ap_df.groupby('cluster')['rating'].mean().reset_index()
              cluster_avg_rating.columns = ['cluster', 'avg_rating']
              ap_df = pd.merge(ap_df, cluster_avg_rating, on='cluster', how='left')


              fig = px.scatter_mapbox(ap_df, 
                                  lat="latitude", 
                                  lon="longitude", 
                                  color="cluster_label", 
                                  hover_name="name", 
                                  hover_data={"cluster": False, "rating": True, "avg_rating": True, "cluster_label": False},
                                  zoom=10, 
                                  mapbox_style="carto-positron",
                                  title="Map for AP Clustering")
              
              fig.update_layout(mapbox=dict(
                  center=dict(lat=ap_df['latitude'].mean(), lon=ap_df['longitude'].mean()),
              ))
              
              fig.update_traces(marker=dict(size=10, opacity=0.8),
                            hovertemplate="<b>%{hovertext}</b><br>" +
                                          "Cluster: %{customdata[0]}<br>" +
                                          "Rating: %{customdata[1]}<br>" +
                                          "Cluster Average Rating: %{customdata[2]}<br>" +
                                          "<extra>Cluster Label: %{customdata[3]}</extra>")

              cluster_stats = ap_df.groupby('cluster').agg(
                  no_of_reviews=('no_of_reviews', 'sum'),
                  avg_rating=('rating', 'mean')
              ).reset_index()
              fig4 = px.bar(
                  cluster_stats, 
                  x='cluster', 
                  y='no_of_reviews',
                  title='Number of Reviews per Cluster',
                  labels={'no_of_reviews': 'Total Reviews'},
                  hover_data={'avg_rating': ':.2f'}  # Display average rating on hover, formatted to 2 decimal places
                  )    
              st.plotly_chart(fig)
              st.plotly_chart(fig4, use_container_width=True)
              
            else:
              st.write("The 'name' column does not exist in the DataFrame.")
        else:
            st.write(message)
        
        # 4. AHC Clustering
        st.subheader("4. Agglomerative Hierarchical Clustering")
        with st.container():
           st.write("✅ **The most optimal hyperparameter values:**")
           st.write("- n_clusters: 6")
           st.write("- affinity: euclidean")
           st.write("- linkage: ward")
           st.write("✅ **Toggle the widgets below to modify the Hyperparameter Values**")
           # Use columns to display widgets side by side
           col1, col2, col3 = st.columns(3)

           # Slider for number of clusters
           with col1:
              n_clusters = st.slider("Number of Clusters", min_value=2, max_value=9, value=6, step=1)

           # Select linkage criterion
           with col2:
              linkage = st.selectbox("Linkage Criterion", options=['ward', 'complete', 'average', 'single'], index=0)
           
           # Select affinity metric
           with col3:
              if linkage == 'ward':
                  affinity = st.selectbox("Affinity Metric", options=['euclidean'], index=0)
              else:
                  affinity = st.selectbox("Affinity Metric", options=['euclidean', 'l1', 'l2', 'manhattan', 'cosine'], index=0)
           st.write("Note: If the linkage is ward, then the affinity metric must be euclidean.")
        ahc_df = st.session_state.dataframe
        if ahc_df is not None:
            bestahc = AgglomerativeClustering(n_clusters=n_clusters, metric=affinity, linkage=linkage)
            #bestahc = AgglomerativeClustering(n_clusters=6, affinity='euclidean', linkage='ward')
            bestahc.fit(st.session_state.X_scaled)
            ahc_df['cluster'] = bestahc.fit_predict(st.session_state.X_scaled)
            
            
            if 'name' in ahc_df.columns:
             
              def map_cluster_to_color(cluster_number):
                  cluster_colors = {
                      0: {'color': 'Cluster 0', 'label': 'Cluster 0'},
                      1: {'color': 'Cluster 1', 'label': 'Cluster 1'},
                      2: {'color': 'Cluster 2', 'label': 'Cluster 2'},
                      3: {'color': 'Cluster 3', 'label': 'Cluster 3'},
                      4: {'color': 'Cluster 4', 'label': 'Cluster 4'},
                      5: {'color': 'Cluster 5', 'label': 'Cluster 5'},
                      6: {'color': 'Cluster 6', 'label': 'Cluster 6'},
                      7: {'color': 'Cluster 7', 'label': 'Cluster 7'},
                      8: {'color': 'Cluster 8', 'label': 'Cluster 8'}
                  }
                  return cluster_colors[cluster_number]['color']
              
              ahc_df['cluster_label'] = ahc_df['cluster'].map(map_cluster_to_color)
              cluster_labels = {0: 'Cluster 2', 1: 'Cluster 0', 2: 'Cluster 1', 3: 'Cluster 3'}
              ahc_df['custom_cluster_label'] = ahc_df['cluster'].map(cluster_labels)
              
              cluster_avg_rating = ahc_df.groupby('cluster')['rating'].mean().reset_index()
              cluster_avg_rating.columns = ['cluster', 'avg_rating']
              ahc_df = pd.merge(ahc_df, cluster_avg_rating, on='cluster', how='left')


              fig = px.scatter_mapbox(ahc_df, 
                                 lat="latitude", 
                                 lon="longitude", 
                                 color="cluster_label", 
                                 hover_name="name", 
                                 hover_data={"cluster": False, "rating": True, "avg_rating": True, "cluster_label": False},
                                 zoom=10, 
                                 mapbox_style="carto-positron",
                                 title="Map for AHC Clustering")
              
              fig.update_layout(mapbox=dict(
                  center=dict(lat=ahc_df['latitude'].mean(), lon=ahc_df['longitude'].mean()),
              ))
              
              fig.update_traces(marker=dict(size=10, opacity=0.8),
                           hovertemplate="<b>%{hovertext}</b><br>" +
                                         "Cluster: %{customdata[0]}<br>" +
                                         "Rating: %{customdata[1]}<br>" +
                                         "Cluster Average Rating: %{customdata[2]}<br>" +
                                         "<extra>Cluster Label: %{customdata[3]}</extra>")

              cluster_stats = ahc_df.groupby('cluster').agg(
                  no_of_reviews=('no_of_reviews', 'sum'),
                  avg_rating=('rating', 'mean')
              ).reset_index()
              fig5 = px.bar(
                  cluster_stats, 
                  x='cluster', 
                  y='no_of_reviews',
                  title='Number of Reviews per Cluster',
                  labels={'no_of_reviews': 'Total Reviews'},
                  hover_data={'avg_rating': ':.2f'}  # Display average rating on hover, formatted to 2 decimal places
                  )    
              st.plotly_chart(fig)
              st.plotly_chart(fig5, use_container_width=True)
         
              
            else:
             st.write("The 'name' column does not exist in the DataFrame.")
        else:
           st.write(message)
        
        # 5. DBSCAN Clustering
        st.subheader("5. DBSCAN Clustering")
        with st.container():
           st.write("✅ **The most optimal hyperparameter values:**")
           st.write("- eps: 0.2")
           st.write("- min_samples: 6")
           st.write("✅ **Toggle the widgets below to modify the Hyperparameter Values**")
           # Use columns to display widgets side by side
           col1, col2 = st.columns(2)

           # Slider for epsilon (eps)
           with col1:
              eps = st.slider("Epsilon (eps)", min_value=0.1, max_value=1.0, value=0.2, step=0.1)

           # Slider for minimum samples
           with col2:
              min_samples = st.slider("Minimum Samples", min_value=1, max_value=20, value=6, step=1)
        db_df = st.session_state.dataframe
        if db_df is not None:
            #bestdb = DBSCAN(eps=0.2, min_samples=6)
            bestdb = DBSCAN(eps=eps, min_samples=min_samples)
            db_df['cluster'] = bestdb.fit_predict(st.session_state.X_scaled)
            
            
            if 'name' in db_df.columns:
              
              def map_cluster_to_color(cluster_number):
                  cluster_colors = {
                      0: {'color': 'Cluster 0', 'label': 'Cluster 0'},
                      1: {'color': 'Cluster 1', 'label': 'Cluster 1'},
                      2: {'color': 'Cluster 2', 'label': 'Cluster 2'},
                      3: {'color': 'Cluster 3', 'label': 'Cluster 3'},
                      4: {'color': 'Cluster 4', 'label': 'Cluster 4'},
                      5: {'color': 'Cluster 5', 'label': 'Cluster 5'},
                      6: {'color': 'Cluster 6', 'label': 'Cluster 5'},
                      7: {'color': 'Cluster 7', 'label': 'Cluster 5'},
                      8: {'color': 'Cluster 8', 'label': 'Cluster 5'},
                      9: {'color': 'Cluster 9', 'label': 'Cluster 5'},
                      10: {'color': 'Cluster 10', 'label': 'Cluster 5'},
                  }
                  
                  default_color = {'color': 'cluster 2', 'label': 'Unknown Cluster'}
                  return cluster_colors.get(cluster_number, default_color)['color']
              
              db_df['cluster_label'] = db_df['cluster'].map(map_cluster_to_color)
              cluster_labels = {0: 'Cluster 2', 1: 'Cluster 0', 2: 'Cluster 1', 3: 'Cluster 3'}
              db_df['custom_cluster_label'] = db_df['cluster'].map(cluster_labels)
              
              cluster_avg_rating = db_df.groupby('cluster')['rating'].mean().reset_index()
              cluster_avg_rating.columns = ['cluster', 'avg_rating']
              db_df = pd.merge(db_df, cluster_avg_rating, on='cluster', how='left')


              fig = px.scatter_mapbox(db_df, 
                                 lat="latitude", 
                                 lon="longitude", 
                                 color="cluster_label", 
                                 hover_name="name", 
                                 hover_data={"cluster": False, "rating": True, "avg_rating": True, "cluster_label": False},
                                 zoom=10, 
                                 mapbox_style="carto-positron",
                                 title="Map for DBSCAN Clustering")
              
              fig.update_layout(mapbox=dict(
                  center=dict(lat=db_df['latitude'].mean(), lon=db_df['longitude'].mean()),
              ))
              
              fig.update_traces(marker=dict(size=10, opacity=0.8),
                           hovertemplate="<b>%{hovertext}</b><br>" +
                                         "Cluster: %{customdata[0]}<br>" +
                                         "Rating: %{customdata[1]}<br>" +
                                         "Cluster Average Rating: %{customdata[2]}<br>" +
                                         "<extra>Cluster Label: %{customdata[3]}</extra>")

              cluster_stats = db_df.groupby('cluster').agg(
                  no_of_reviews=('no_of_reviews', 'sum'),
                  avg_rating=('rating', 'mean')
              ).reset_index()
              fig6 = px.bar(
                  cluster_stats, 
                  x='cluster', 
                  y='no_of_reviews',
                  title='Number of Reviews per Cluster',
                  labels={'no_of_reviews': 'Total Reviews'},
                  hover_data={'avg_rating': ':.2f'}  # Display average rating on hover, formatted to 2 decimal places
                  )    
              st.plotly_chart(fig)
              st.plotly_chart(fig6, use_container_width=True)
         
              
            else:
             st.write("The 'name' column does not exist in the DataFrame.")
        else:
           st.write(message)
           
           
   ############################################################################################        
    elif step == 'Display Rating Plots':
        
        st.write("<h1 style='width: 800px;'>Rating Plots</h1>", unsafe_allow_html=True)
        st.subheader("1. K Means Clustering")
        
        rating = st.slider("Rating", min_value=1, max_value=5, value=3, step=1)
        
            
        kmean_df = st.session_state.dataframe
        if kmean_df is not None:        
                
               filtered_df = kmean_df[kmean_df['rating'] == rating]
                
               bestKM = KMeans(n_clusters=5, random_state=42, n_init = 'auto', init = 'k-means++', max_iter = 300, tol = 0.0001)
               #bestKM = KMeans(n_clusters=n_clusters, random_state=42, n_init = 'auto', init = init_method, max_iter = max_iter, tol = tolerance)
               bestKM.fit(filtered_df[['latitude', 'longitude']])
               cluster_labels = bestKM.labels_          
               
               filtered_df.loc[:, 'cluster'] = cluster_labels
               
               if 'name' in kmean_df.columns:
                 
                 def map_cluster_to_color(cluster_number):
                     cluster_colors = {
                         0: {'color': 'Cluster 0', 'label': 'Cluster 0'},
                         1: {'color': 'Cluster 1', 'label': 'Cluster 1'},
                         2: {'color': 'Cluster 2', 'label': 'Cluster 2'},
                         3: {'color': 'Cluster 3', 'label': 'Cluster 3'},
                         4: {'color': 'Cluster 4', 'label': 'Cluster 4'}
                     }
                     return cluster_colors[cluster_number]['color']
                 
                 filtered_df['cluster_label'] = filtered_df['cluster'].map(map_cluster_to_color)
                 cluster_labels = {0: 'Cluster 2', 1: 'Cluster 0', 2: 'Cluster 1', 3: 'Cluster 3'}
                 filtered_df['custom_cluster_label'] = filtered_df['cluster'].map(cluster_labels)
                 
                 cluster_avg_rating = filtered_df.groupby('cluster')['rating'].mean().reset_index()
                 cluster_avg_rating.columns = ['cluster', 'avg_rating']
                 filtered_df = pd.merge(filtered_df, cluster_avg_rating, on='cluster', how='left')


                 fig = px.scatter_mapbox(filtered_df, 
                                    lat="latitude", 
                                    lon="longitude", 
                                    color="cluster_label", 
                                    hover_name="name", 
                                    hover_data={"cluster": False, "rating": True, "avg_rating": True, "cluster_label": False},
                                    zoom=10, 
                                    mapbox_style="carto-positron",
                                    title="Map for K-Means Clustering")
                 
                 fig.update_layout(mapbox=dict(
                     center=dict(lat=filtered_df['latitude'].mean(), lon=filtered_df['longitude'].mean()),
                 ))
                 
                 fig.update_traces(marker=dict(size=10, opacity=0.8),
                              hovertemplate="<b>%{hovertext}</b><br>" +
                                            "Cluster: %{customdata[0]}<br>" +
                                            "Rating: %{customdata[1]}<br>" +
                                            "<extra>Cluster Label: %{customdata[3]}</extra>")

                 st.plotly_chart(fig)

               else:
                  st.write("The 'name' column does not exist in the DataFrame.")
        else:
                st.write("Nothing to display here. Please upload a dataset. Thank you.")
                
        ###################################################################################
                
        st.subheader("2. Birch Clustering")
        
        rating = st.slider("Rating", min_value=1, max_value=5, value=3, step=1, key="birch_rating_slider")
        
        birch_df = st.session_state.dataframe
        if birch_df is not None:
            
            filtered_df = birch_df[birch_df['rating'] == rating]
            
            bestbirch = Birch(branching_factor = 25, n_clusters = 4, threshold = 0.1)
            #bestbirch = Birch(branching_factor=branching_factor, n_clusters=n_clusters, threshold=threshold)
            bestbirch.fit(filtered_df[['latitude', 'longitude']])
            cluster_labels = bestbirch.labels_          
            
            filtered_df.loc[:, 'cluster'] = cluster_labels
            
            

            if 'name' in birch_df.columns:              
              def map_cluster_to_color(cluster_number):
                  cluster_colors = {
                      0: {'color': 'Cluster 0', 'label': 'Cluster 0'},
                      1: {'color': 'Cluster 1', 'label': 'Cluster 1'},
                      2: {'color': 'Cluster 2', 'label': 'Cluster 2'},
                      3: {'color': 'Cluster 3', 'label': 'Cluster 3'}
                  }
                  return cluster_colors[cluster_number]['color']
              
              filtered_df['cluster_label'] = filtered_df['cluster'].map(map_cluster_to_color)
              cluster_labels = {0: 'Cluster 2', 1: 'Cluster 0', 2: 'Cluster 1', 3: 'Cluster 3'}
              filtered_df['custom_cluster_label'] = filtered_df['cluster'].map(cluster_labels)
              
              cluster_avg_rating = filtered_df.groupby('cluster')['rating'].mean().reset_index()
              cluster_avg_rating.columns = ['cluster', 'avg_rating']
              filtered_df = pd.merge(filtered_df, cluster_avg_rating, on='cluster', how='left')


              fig = px.scatter_mapbox(filtered_df, 
                                 lat="latitude", 
                                 lon="longitude", 
                                 color="cluster_label", 
                                 hover_name="name", 
                                 hover_data={"cluster": False, "rating": True, "avg_rating": True, "cluster_label": False},
                                 zoom=10, 
                                 mapbox_style="carto-positron",
                                 title="Map for Birch Clustering")
              
              fig.update_layout(mapbox=dict(
                  center=dict(lat=filtered_df['latitude'].mean(), lon=filtered_df['longitude'].mean()),
              ))
              
              fig.update_traces(marker=dict(size=10, opacity=0.8),
                           hovertemplate="<b>%{hovertext}</b><br>" +
                                         "Cluster: %{customdata[0]}<br>" +
                                         "Rating: %{customdata[1]}<br>" +
                                         "<extra>Cluster Label: %{customdata[3]}</extra>")

              st.plotly_chart(fig)
         
              
            else:
             st.write("The 'name' column does not exist in the DataFrame.")
        else:
           st.write("Nothing to display here. Please upload a dataset. Thank you.")
           
          ###################################################################
    
        st.subheader("3. Affinity Propagation Clustering")
        
        rating = st.slider("Rating", min_value=1, max_value=5, value=3, step=1, key="ap_rating_slider")
        
        ap_df = st.session_state.dataframe
        if ap_df is not None:
            
            filtered_df = ap_df[ap_df['rating'] == rating]
            #bestap = AffinityPropagation(affinity='euclidean', damping=damping, preference=preference, random_state = None, verbose = False, max_iter=200, copy=True, convergence_iter=15).fit(st.session_state.X_scaled)
            bestap = AffinityPropagation(affinity='euclidean', damping=0.7, preference=-5, random_state = None, verbose = False, max_iter=200, copy=True, convergence_iter=15).fit(st.session_state.X_scaled)
            bestap.fit(filtered_df[['latitude', 'longitude']])
            
            cluster_labels = bestap.labels_          
            
            filtered_df.loc[:, 'cluster'] = cluster_labels
            
            if 'name' in ap_df.columns:
              
              def map_cluster_to_color(cluster_number):
                  cluster_colors = {
                      0: {'color': 'Cluster 0', 'label': 'Cluster 0'},
                      1: {'color': 'Cluster 1', 'label': 'Cluster 1'},
                      2: {'color': 'Cluster 2', 'label': 'Cluster 2'},
                      3: {'color': 'Cluster 3', 'label': 'Cluster 3'},
                      4: {'color': 'Cluster 4', 'label': 'Cluster 4'},
                      5: {'color': 'Cluster 5', 'label': 'Cluster 5'},
                      6: {'color': 'Cluster 6', 'label': 'Cluster 6'},
                      7: {'color': 'Cluster 7', 'label': 'Cluster 7'},
                      8: {'color': 'Cluster 8', 'label': 'Cluster 8'}
                  }
                  return cluster_colors[cluster_number]['color']
              
              filtered_df['cluster_label'] = filtered_df['cluster'].map(map_cluster_to_color)
              cluster_labels = {0: 'Cluster 2', 1: 'Cluster 0', 2: 'Cluster 1', 3: 'Cluster 3'}
              filtered_df['custom_cluster_label'] = filtered_df['cluster'].map(cluster_labels)
              
              cluster_avg_rating = filtered_df.groupby('cluster')['rating'].mean().reset_index()
              cluster_avg_rating.columns = ['cluster', 'avg_rating']
              filtered_df = pd.merge(filtered_df, cluster_avg_rating, on='cluster', how='left')


              fig = px.scatter_mapbox(filtered_df, 
                                  lat="latitude", 
                                  lon="longitude", 
                                  color="cluster_label", 
                                  hover_name="name", 
                                  hover_data={"cluster": False, "rating": True, "avg_rating": True, "cluster_label": False},
                                  zoom=10, 
                                  mapbox_style="carto-positron",
                                  title="Map for AP Clustering")
              
              fig.update_layout(mapbox=dict(
                  center=dict(lat=filtered_df['latitude'].mean(), lon=filtered_df['longitude'].mean()),
              ))
              
              fig.update_traces(marker=dict(size=10, opacity=0.8),
                            hovertemplate="<b>%{hovertext}</b><br>" +
                                          "Cluster: %{customdata[0]}<br>" +
                                          "Rating: %{customdata[1]}<br>" +
                                          "<extra>Cluster Label: %{customdata[3]}</extra>")

              st.plotly_chart(fig)
         
              
            else:
              st.write("The 'name' column does not exist in the DataFrame.")
        else:
            st.write("Nothing to display here. Please upload a dataset. Thank you.")
        
        #########################################################################
            
        st.subheader("4. Agglomerative Hierarchical Clustering")
        
        rating = st.slider("Rating", min_value=1, max_value=5, value=3, step=1, key="ahc_rating_slider")
        
        ahc_df = st.session_state.dataframe
        if ahc_df is not None:
            filtered_df = ahc_df[ahc_df['rating'] == rating]
            #bestahc = AgglomerativeClustering(n_clusters=n_clusters, metric=affinity, linkage=linkage)
            #got problem
            bestahc = AgglomerativeClustering(n_clusters=6, linkage='ward')
            bestahc.fit(filtered_df[['latitude', 'longitude']])
            
            cluster_labels = bestahc.labels_ 
            filtered_df.loc[:, 'cluster'] = cluster_labels
            
            
            if 'name' in ahc_df.columns:
             
              def map_cluster_to_color(cluster_number):
                  cluster_colors = {
                      0: {'color': 'Cluster 0', 'label': 'Cluster 0'},
                      1: {'color': 'Cluster 1', 'label': 'Cluster 1'},
                      2: {'color': 'Cluster 2', 'label': 'Cluster 2'},
                      3: {'color': 'Cluster 3', 'label': 'Cluster 3'},
                      4: {'color': 'Cluster 4', 'label': 'Cluster 4'},
                      5: {'color': 'Cluster 5', 'label': 'Cluster 5'},
                      6: {'color': 'Cluster 6', 'label': 'Cluster 6'},
                      7: {'color': 'Cluster 7', 'label': 'Cluster 7'},
                      8: {'color': 'Cluster 8', 'label': 'Cluster 8'}
                  }
                  return cluster_colors[cluster_number]['color']
              
              filtered_df['cluster_label'] = filtered_df['cluster'].map(map_cluster_to_color)
              cluster_labels = {0: 'Cluster 2', 1: 'Cluster 0', 2: 'Cluster 1', 3: 'Cluster 3'}
              filtered_df['custom_cluster_label'] = filtered_df['cluster'].map(cluster_labels)
              
              cluster_avg_rating = filtered_df.groupby('cluster')['rating'].mean().reset_index()
              cluster_avg_rating.columns = ['cluster', 'avg_rating']
              filtered_df = pd.merge(filtered_df, cluster_avg_rating, on='cluster', how='left')


              fig = px.scatter_mapbox(filtered_df, 
                                 lat="latitude", 
                                 lon="longitude", 
                                 color="cluster_label", 
                                 hover_name="name", 
                                 hover_data={"cluster": False, "rating": True, "avg_rating": True, "cluster_label": False},
                                 zoom=10, 
                                 mapbox_style="carto-positron",
                                 title="Map for AHC Clustering")
              
              fig.update_layout(mapbox=dict(
                  center=dict(lat=filtered_df['latitude'].mean(), lon=filtered_df['longitude'].mean()),
              ))
              
              fig.update_traces(marker=dict(size=10, opacity=0.8),
                           hovertemplate="<b>%{hovertext}</b><br>" +
                                         "Cluster: %{customdata[0]}<br>" +
                                         "Rating: %{customdata[1]}<br>" +
                                         "<extra>Cluster Label: %{customdata[3]}</extra>")

              st.plotly_chart(fig)
         
              
            else:
             st.write("The 'name' column does not exist in the DataFrame.")
        else:
           st.write("Nothing to display here. Please upload a dataset. Thank you.")
           
        st.subheader("5. DBSCAN Clustering")
        
        rating = st.slider("Rating", min_value=1, max_value=5, value=3, step=1, key="db_rating_slider")
        
        db_df = st.session_state.dataframe
        
        if db_df is not None:
            filtered_df = db_df[db_df['rating'] == rating]
            bestdb = DBSCAN(eps=0.2, min_samples=6)
            #bestdb = DBSCAN(eps=eps, min_samples=min_samples)
            bestdb.fit(filtered_df[['latitude', 'longitude']])
            
            cluster_labels = bestdb.labels_ 
            filtered_df.loc[:, 'cluster'] = cluster_labels
            
            
            if 'name' in db_df.columns:
              
              def map_cluster_to_color(cluster_number):
                  cluster_colors = {
                      0: {'color': 'Cluster 0', 'label': 'Cluster 0'},
                      1: {'color': 'Cluster 1', 'label': 'Cluster 1'},
                      2: {'color': 'Cluster 2', 'label': 'Cluster 2'},
                      3: {'color': 'Cluster 3', 'label': 'Cluster 3'},
                      4: {'color': 'Cluster 4', 'label': 'Cluster 4'},
                      5: {'color': 'Cluster 5', 'label': 'Cluster 5'},
                      6: {'color': 'Cluster 6', 'label': 'Cluster 5'},
                      7: {'color': 'Cluster 7', 'label': 'Cluster 5'},
                      8: {'color': 'Cluster 8', 'label': 'Cluster 5'},
                      9: {'color': 'Cluster 9', 'label': 'Cluster 5'},
                      10: {'color': 'Cluster 10', 'label': 'Cluster 5'},
                  }
                  
                  default_color = {'color': 'cluster 2', 'label': 'Unknown Cluster'}
                  return cluster_colors.get(cluster_number, default_color)['color']
              
              filtered_df['cluster_label'] = filtered_df['cluster'].map(map_cluster_to_color)
              cluster_labels = {0: 'Cluster 2', 1: 'Cluster 0', 2: 'Cluster 1', 3: 'Cluster 3'}
              filtered_df['custom_cluster_label'] = filtered_df['cluster'].map(cluster_labels)
              
              cluster_avg_rating = filtered_df.groupby('cluster')['rating'].mean().reset_index()
              cluster_avg_rating.columns = ['cluster', 'avg_rating']
              filtered_df = pd.merge(filtered_df, cluster_avg_rating, on='cluster', how='left')


              fig = px.scatter_mapbox(filtered_df, 
                                 lat="latitude", 
                                 lon="longitude", 
                                 color="cluster_label", 
                                 hover_name="name", 
                                 hover_data={"cluster": False, "rating": True, "avg_rating": True, "cluster_label": False},
                                 zoom=10, 
                                 mapbox_style="carto-positron",
                                 title="Map for DBSCAN Clustering")
              
              fig.update_layout(mapbox=dict(
                  center=dict(lat=filtered_df['latitude'].mean(), lon=filtered_df['longitude'].mean()),
              ))
              
              fig.update_traces(marker=dict(size=10, opacity=0.8),
                           hovertemplate="<b>%{hovertext}</b><br>" +
                                         "Cluster: %{customdata[0]}<br>" +
                                         "Rating: %{customdata[1]}<br>" +
                                         "<extra>Cluster Label: %{customdata[3]}</extra>")

              st.plotly_chart(fig)
         
              
            else:
             st.write("The 'name' column does not exist in the DataFrame.")
        else:
           st.write("Nothing to display here. Please upload a dataset. Thank you.")
    ########################################################################################################
    elif step == "Select Suburb or Road":
        st.write("<h1 style='width: 800px;'>Suburb or Road of Western Restaurants in Singapore</h1>", unsafe_allow_html=True)
        if st.session_state.dataframe is not None:
            df = st.session_state.dataframe
            suburb_options = ['Please select an option'] + list(df['suburb'].unique())
            selected_suburb = st.selectbox("Select Suburb:", suburb_options, key="suburb_selectbox")

            if selected_suburb != 'Please select an option':
                # Filter dataframe for selected suburb
                filtered_df_suburb = df[df['suburb'] == selected_suburb]
                # Get unique roads in the selected suburb
                road_options = ['Please select an option'] + list(filtered_df_suburb['road'].unique())
                
                selected_road = st.selectbox("Select Road:", road_options, key="road_selectbox")

                if selected_road != 'Please select an option':
                    st.write("Suburb selected:", selected_suburb)
                    st.write("Road selected:", selected_road)

                    # Combine restaurants in the selected suburb and road
                    st.subheader("Restaurants in selected suburb and road:")
                    filtered_df_combined = df[(df['suburb'] == selected_suburb) & (df['road'] == selected_road)]
                    st.write(filtered_df_combined)
                    
                    # Display restaurants in the selected suburb
                    st.subheader("Restaurants in selected suburb:")
                    filtered_df_suburb = df[df['suburb'] == selected_suburb]
                    st.write(filtered_df_suburb)

                    # Display restaurants in the selected road
                    st.subheader("Restaurants in selected road:")
                    filtered_df_road = df[df['road'] == selected_road]
                    st.write(filtered_df_road)       
        else:
           st.write("Please upload a dataset.")

if __name__ == "__main__":
    main()