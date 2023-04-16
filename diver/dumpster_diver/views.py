from spotify.spotify_wrapper import SpotifyWrapper
from models.msd_model import MSDModel
from django.shortcuts import render
import os
import pandas as pd
import logging
import copy


# Create your views here.
def index(request):
    # Set up logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Define the number of recos we will provide per song
    recos_per_song = 3
        
    # Initialize the SpotifyWrapper
    logger.info("Retrieving User Spotify Songs")
    wrapper = SpotifyWrapper()
    
    # Fetch user's recent tracks and features
    user_tracks_df, user_dumpster_diver_features_df = wrapper.get_user_recent_tracks(top_tracks_limit=5)

    # Define parameters for Spotify recommendations
    spotify_recs_list_dfs = []
    spotify_recs_list_dumpster_features_dfs = []
    for i in range(len(user_tracks_df)):
        seed_tracks = [user_tracks_df.iloc[i]["track_id"], ]

        # Generate our params for recommendations
        sample_params = {
            "limit": recos_per_song,
            "seed_tracks": seed_tracks,
        }

        # Get Spotify recommendations and clean up the dataframe
        cur_rec_df, cur_dumpster_features_df = wrapper.get_spotify_recommendations(**sample_params)
        # TODO: Compute all the metrics for the spotify recos as our MSD features
        spotify_recs_list_dfs.append(cur_rec_df)
        spotify_recs_list_dumpster_features_dfs.append(cur_dumpster_features_df)

    spotify_recs_df = pd.concat(spotify_recs_list_dfs).reset_index()
    spotify_recs_dumpster_features_df = pd.concat(spotify_recs_list_dumpster_features_dfs).reset_index()

    # Clean the dumpster diver features
    rename_columns_dict = {
        'name': 'Song Title',
        'artist': 'Artist',
        'popularity': 'Popularity',
        'loudness': 'Loudness'
    }
    cleaned_spotify_recs_df = clean_dataframe(spotify_recs_df, tracks=True, rename_columns=rename_columns_dict)

    #TODO: Add the spotify recs to the PCA plot

    # Load the MSDModel
    df_filepath = "spotify/msd.pkl"
    logger.info("Loading the MSD Model: {}".format(df_filepath))
    df = pd.read_pickle(df_filepath)
    msd_model = MSDModel(df)

    # Get the MSD recommendations and transform data
    logger.info("Diving into the dumpster!")
    msd_recs_list_dfs = []
    msd_recs_list_dumpster_features = []
    for i in range(len(user_dumpster_diver_features_df)):
        cur_rec_df, cur_dumpster_features_df = msd_model.find_k_neighbors(
            user_dumpster_diver_features_df.iloc[i:i+1],
            n_neighbors=recos_per_song
        )
        msd_recs_list_dfs.append(cur_rec_df)
        msd_recs_list_dumpster_features.append(cur_dumpster_features_df)
    
    msd_recs_df = pd.concat(msd_recs_list_dfs).reset_index()

    # Clean up the MSD recommendations and user's recent tracks data
    rename_columns_dict = {
        'artist_name': 'Artist',
        'song_title': 'Song Title',
        'song_hotttnesss': 'Popularity',
        'loudness': 'Loudness'
    }
    clean_msd_rec_df = clean_dataframe(msd_recs_df, rename_columns=rename_columns_dict)
    rename_columns_dict = {
        'name': 'Song Title',
        'artist': 'Artist',
        'popularity': 'Popularity',
        'loudness': 'Loudness'
    }
    clean_tracks_df = clean_dataframe(user_tracks_df.drop(columns=['song_array']), tracks=True, rename_columns=rename_columns_dict)

    # TODO: Plots and plots and plots
    # logger.info("Displaying our Dumpster Finds!")
    # msd_plot = wrapper.plot_msd()
    # features, parallel_cords, features_merged, parallel_cords_merged  = wrapper.plot_song_data(tracks_df, recommendations_df)

    # Prepare the data to be passed to the frontend
    context = {
        ## Put data here that you want to pass to the frontend in key-value pair/dictionary form:
        ## 'key':variable,
        'recommendations': clean_msd_rec_df.to_html(classes='table table-bordered table-striped table-dark table-hover', table_id='rec-table', index=False),
        'spotify_recs': cleaned_spotify_recs_df.to_html(classes='table table-bordered table-striped table-dark table-hover', table_id='rec-table', index=False),
        'tracks': clean_tracks_df.to_html(classes='table table-bordered table-striped table-dark table-hover', table_id='tracks-table', index=False),
        # 'msd_plot': msd_plot,
        # 'features': features,
        # 'parallel_cords': parallel_cords,
        # 'features_merged': features_merged,
        # 'parallel_cords_merged': parallel_cords_merged,
        # 'pitch_network': pitch_network_df.to_html()
    }

    return render(request, 'dumpster_diver/index.html', context)

def clean_dataframe(df, tracks=False, rename_columns=None):
    drop_columns = ['index', 'msd_id', 'artist_id', 'artist_familiarity', 'artist_hotttnesss', 'song_id',
                    'year', 'energy', 'danceability', 'tempo', 'pitch_network_average_degree',
                    'pitch_network_entropy', 'pitch_network_mean_clustering_coeff', 'timbre_00', 'timbre_01',
                    'timbre_02', 'timbre_03', 'timbre_04', 'timbre_05', 'timbre_06', 'timbre_07', 'timbre_08',
                    'timbre_09', 'timbre_10', 'timbre_11']

    cleaned_df = df.drop(labels=[col for col in drop_columns if col in df.columns], axis=1)

    if rename_columns:
        cleaned_df.rename(columns=rename_columns, inplace=True)

    if tracks:
        cleaned_df['Popularity'] = cleaned_df['Popularity'] / 100

    # Round the Popularity column to 2 decimal places
    cleaned_df['Popularity'] = cleaned_df['Popularity'].round(2)

    cleaned_df = cleaned_df[['Song Title', 'Artist', 'Popularity']]

    # Remove any profanity
    if os.environ.get("DUMPSTER_DIVER_CENSOR", "False") == "True":
        from pdb import set_trace; set_trace()
        for col in ['Song Title', 'Artist']:
            from better_profanity import profanity
            cleaned_df[col] = cleaned_df[col].map(lambda x: profanity.censor(x))

    return cleaned_df
