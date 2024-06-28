from pathlib import Path
import numpy as np
import pandas as pd
import spotipy
import yaml
import time
from typing import Dict, Tuple, List
from database_generation.pitch_network import compute_pitch_network_stats, create_pitch_network
from spotipy.oauth2 import SpotifyOAuth
from pathlib import Path
import plotly.express as px
import plotly.offline as opy
from sklearn.decomposition import PCA


class SpotifyWrapper:
    def __init__(self):
        # Read credentials from config.yml
        # spotify:
        #    id: "<id>"
        #    client_id: "<client_id>" 
        #    client_secret: "<client_secret>"
        #    redirect_uri: "http://localhost:8888/callback" <-- Make sure to add this to your Spotify Client!

        config_file = Path(__file__).parent / "config.yml"

        with open(config_file, 'r') as f:
            self.config = yaml.safe_load(f)

        self.client_id = self.config["spotify"]["client_id"]
        self.client_secret = self.config["spotify"]["client_secret"]
        self.redirect_uri = self.config["spotify"]["redirect_uri"]

        # Set the required scopes for access
        scope = "user-read-recently-played user-library-read user-read-private user-read-email user-top-read"

        # Authenticate with Spotify
        self.sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=self.client_id,
                                                            client_secret=self.client_secret,
                                                            redirect_uri=self.redirect_uri,
                                                            scope=scope))

#    def get_user_recent_tracks(self, limit: int = 5) -> pd.DataFrame:
    def get_user_recent_tracks(self, top_tracks_limit: int = 5, recent_tracks_limit: int = 0) -> Tuple[pd.DataFrame, pd.DataFrame]:

        """
        Retrieves the recently played songs by a user with song features

        :param top_tracks_limit: The number of top songs to return, DEFAULT of 5.
        :param recent_tracks_limit: The number of recent songs to return, DEFAULT of 5.
        :returns: DataFrame of the songs and their features
        """

        # Get the user's recently played and top songs
        #recent_tracks = self.sp.current_user_recently_played(limit=recent_tracks_limit)
        top_tracks = self.sp.current_user_top_tracks(limit=top_tracks_limit)

        # Parse and return the dataframes
        return self._generate_dfs([{'track': item} for item in top_tracks['items']])

    def get_spotify_recommendations(self,
                                    seed_artists=None, seed_genres=None, seed_tracks=None,
                                    limit=20, market=None,
                                    target_acousticness=None, target_duration_ms=None, target_instrumentalness=None,
                                    target_key=None, target_liveness=None, target_loudness=None, target_mode=None,
                                    target_popularity=None, target_speechiness=None, target_tempo=None,
                                    target_time_signature=None, target_valence=None, max_popularity=None,
                                    sleep_time=10,
                                    ) -> Tuple[pd.DataFrame, pd.DataFrame]:

        params = {
            "limit": limit,
            "seed_artists": seed_artists if seed_artists else None,
            "seed_genres": seed_genres if seed_genres else None,
            "seed_tracks": seed_tracks if seed_tracks else None,
            "market": market,
            "target_acousticness": target_acousticness,
            "target_duration_ms": target_duration_ms,
            "target_instrumentalness": target_instrumentalness,
            "target_key": target_key,
            "target_liveness": target_liveness,
            "target_loudness": target_loudness,
            "target_mode": target_mode,
            "target_popularity": target_popularity,
            "target_speechiness": target_speechiness,
            "target_tempo": target_tempo,
            "target_time_signature": target_time_signature,
            "target_valence": target_valence,
            "max_popularity": max_popularity
        }

        # Remove None values from the params dictionary
        params = {k: v for k, v in params.items() if v is not None}
        recommendations = None

        # Loop until we get a recommendation
        while recommendations is None:
            recommendations = self.sp.recommendations(**params)
            if len(recommendations["tracks"]) == 0:
                time.sleep(sleep_time)
                recommendations = None
                
        return self._generate_dfs([{'track': item} for item in recommendations["tracks"]])
        
        
    def _generate_dfs(self, spotify_tracks: List[Dict]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Extract track IDs
        track_ids = [track['track']['id'] for track in spotify_tracks]

        # Get the track details including popularity
        track_details = [self.sp.track(track_id) for track_id in track_ids]

        # Get the audio features of the songs
        audio_features = self.sp.audio_features(track_ids)

        # Get the song array data
        song_arrays = [self.sp.audio_analysis(track_id)['segments'] for track_id in track_ids]

        # Compute the pitch network stats and create the pitch network
        pitch_network_data_list = []
        for song in song_arrays:
            pitch_array = np.array([segment['pitches'] for segment in song])
            pitch_network, pitch_codewords = create_pitch_network(pitch_array)
            average_degree, graph_entropy, average_clustering = compute_pitch_network_stats(pitch_array)
            pitch_network_data_list.append({
                'average_degree': average_degree,
                'graph_entropy': graph_entropy,
                'average_clustering': average_clustering,
                'pitch_network': pitch_network,
                'pitch_codewords': pitch_codewords
            })

        # Compute the average timbre values for each track
        average_timbre_values = []
        for song in song_arrays:
            timbre_array = np.array([segment['timbre'] for segment in song])
            average_timbre = np.mean(timbre_array, axis=0)
            average_timbre_values.append(average_timbre)

        # Get the genres of the songs
        track_genres = []
        for track_id in track_ids:
            track = self.sp.track(track_id)
            artist_id = track['artists'][0]['id']
            artist = self.sp.artist(artist_id)
            genres = artist.get('genres')
            track_genres.append(genres)

        # Combine track information and audio features
        combined_data = []
        dumpster_diver_features = []

        for track, details, features, song_array, genres, pitch_stats in \
                zip(spotify_tracks, track_details, audio_features, song_arrays, track_genres, pitch_network_data_list):
            track_info = {
                'track_id': track['track']['id'],
                'name': track['track']['name'],
                'artist': track['track']['artists'][0]['name'],
                'artist_id': track['track']['artists'][0]['id'],
                'played_at': track.get('played_at', None),
                'popularity': details['popularity'],
                'song_array': song_array,
                'genres': genres
            }
            combined_data.append({**track_info, **features})

            dumpster_diver_features_dict = {
                'track_id': track['track']['id'],
                'name': track['track']['name'],
                'artist': track['track']['artists'][0]['name'],
                'loudness': features['loudness'],
                'tempo': features['tempo'],
                'pitch_network_average_degree': pitch_stats['average_degree'],
                'pitch_network_entropy': pitch_stats['graph_entropy'],
                'pitch_network_mean_clustering_coeff': pitch_stats['average_clustering'],
                'timbre_00': average_timbre[0],
                'timbre_01': average_timbre[1],
                'timbre_02': average_timbre[2],
                'timbre_03': average_timbre[3],
                'timbre_04': average_timbre[4],
                'timbre_05': average_timbre[5],
                'timbre_06': average_timbre[6],
                'timbre_07': average_timbre[7],
                'timbre_08': average_timbre[8],
                'timbre_09': average_timbre[9],
                'timbre_10': average_timbre[10],
                'timbre_11': average_timbre[11],
                'pitch_network': pitch_stats['pitch_network'],
                'pitch_codewords': pitch_stats['pitch_codewords']
            }

            # Update pitch network data dictionary with pitch_stats
            dumpster_diver_features_dict.update(pitch_stats)
            dumpster_diver_features.append(dumpster_diver_features_dict)

        # Store everything into a dataframe
        df = pd.DataFrame(combined_data)

        # Create pitch_network DataFrame
        dumpster_diver_features_df = pd.DataFrame(dumpster_diver_features)

        return df, dumpster_diver_features_df
    

    def plot_song_data(self, df_songs, df_recs):
        if df_recs.empty:
            print("""""\nXXXXXXXXXXXXXXXX\n
            NO RECCOMENDATAIONS --- PASSING
            \nXXXXXXXXXXXXXXXX\n""")
            df_recs = df_songs
        print(f'Reccomended songs df backend = {df_recs.head()}')

        df_songs["tempo_normalized"] = df_songs["tempo"] / df_songs["tempo"].max()
        df_songs["popularity_normalized"] = df_songs["popularity"] / 100

        df_recs["tempo_normalized"] = df_recs["tempo"] / df_recs["tempo"].max()
        df_recs["popularity_normalized"] = df_recs["popularity"] / 100

        features = ["tempo_normalized", "popularity_normalized", "energy", "danceability","key"]

        fig = px.scatter_matrix(
            df_songs,
            dimensions=features,
            color="key",
            hover_name="name",
            template="plotly_dark",
            labels={"tempo_normalized": "Tempo",
                  "popularity_normalized": "Popularity Width", "energy": "Energy",
                  "danceability": "Danceability", "key": "Key", }
        )
        div = opy.plot(fig, auto_open=False, output_type='div')

        fig = px.parallel_coordinates(df_songs[features], 
                                    color="key",
                                    color_continuous_midpoint=3,
                                    template="plotly_dark",
                                    labels={"tempo_normalized": "Tempo",
                                            "popularity_normalized": "Popularity", "energy": "Energy",
                                            "danceability": "Danceability", "key": "Key", }
        )
        div2 = opy.plot(fig, auto_open=False, output_type='div')

        merged_df = pd.concat([df_songs, df_recs], axis=0, keys=['from_songs','from_recs']).reset_index(level=[0])
        features.append("level_0")
        print(f'merged_df {merged_df.head()}')
        fig = px.scatter_matrix(
            merged_df,
            dimensions=features,
            color="level_0",
            hover_name="name",
            template="plotly_dark",
            labels={"tempo_normalized": "Tempo",
                  "popularity_normalized": "Popularity Width", "energy": "Energy",
                  "danceability": "Danceability", "key": "Key", "level_0": "Source", }
        )
        div3 = opy.plot(fig, auto_open=False, output_type='div')

        fig = px.parallel_coordinates(merged_df[features], 
                                    color="key",
                                    color_continuous_midpoint=3,
                                    template="plotly_dark",
                                    labels={"tempo_normalized": "Tempo",
                                            "popularity_normalized": "Popularity", "energy": "Energy",
                                            "danceability": "Danceability", "key": "Key", "level_0": "Source",}
        )
        div4 = opy.plot(fig, auto_open=False, output_type='div')

        return div, div2, div3, div4
    
    def plot_msd(self):

        # Replace the pkl file location below with public-facing URL, as needed
        df = pd.read_pickle("spotify/msd.pkl")
        # print('msd_df')
        # print(df)

        # for col in df.columns:
        #     print(col)
        # features = list(df.columns[8:])

        # pca = PCA(n_components=3)
        # components = pca.fit_transform(df[features])
        # # fig = px.scatter_3d(
        # #     components,
        # #     x=0,
        # #     y=1,
        # #     z=2,
        # #     color=df["year"],
        # #     hover_name="song_title",
        # #     template="plotly_dark"
        # # )
        # total_var = pca.explained_variance_ratio_.sum() * 100
        # fig = px.scatter_3d(
        #     components, x=0, y=1, z=2, color=df['year'],
        #     title=f'Total Explained Variance: {total_var:.2f}%',
        #     labels={'0': 'PC 1', '1': 'PC 2', '2': 'PC 3'},
        #     hover_name=df["song_title"],
        #     template="plotly_dark",
        #     height=1200
        # )       

        # # fig = px.scatter_matrix(
        # #     df.head(),
        # #     dimensions=features,
        # #     color="tempo",
        # #     hover_name="song_title",
        # #     template="plotly_dark"
        # # )

        # # fig = px.scatter_3d(df.sample(10000), x='pitch_network_average_degree', y='pitch_network_entropy', z='pitch_network_mean_clustering_coeff',
        # #             color="tempo",
        # #             template="plotly_dark",
        # #             height=1200, 
        # #             hover_name="song_title",
        # #             animation_frame="year")
        # # fig.update_layout(
        # # scene = dict(
        # #     xaxis = dict(range=[0,13],),
        # #                 yaxis = dict(range=[0,1.75],),
        # #                 zaxis = dict(range=[0,1],),)
        # # )
        # div = opy.plot(fig, auto_open=False, output_type='div')

        # # fig = px.scatter_3d(df, x='pitch_network_average_degree', y='pitch_network_entropy', z='pitch_network_mean_clustering_coeff',
        # #             color="tempo",
        # #             template="plotly_dark",
        # #             height=1200, 
        # #             hover_name="song_title")
        # # div = opy.plot(fig, auto_open=False, output_type='div')

        # # fig = px.scatter_matrix(
        # #     df.sample(1000),
        # #     dimensions=features,
        # #     color="year",
        # #     template="plotly_dark",
        # #     height=1200,
        # #     hover_name="song_title"
        # # )
        # # df["year"] = df["year"].astype(str)
        # # fig = px.parallel_coordinates(df[features].sample(1000), 
        # #                             color="year",
        # #                             template="plotly_dark",
        # #                             height=1200
        # #                             )

        # # fig = px.scatter(df, x="pitch_network_average_degree", y="pitch_network_entropy", animation_frame="year",
        # #                 color="tempo",
        # #                 hover_name="song_title",
        # #                 template="plotly_dark",
        # #                 height=1200
        # # )
        # div = opy.plot(fig, auto_open=False, output_type='div')

        # return div
    def get_audio_analysis(self, x):
        audio_analysis = self.sp.audio_analysis(x["track_id"])
        pitches = []
        timbres = []
        for i in range(len(audio_analysis["segments"])):
            pitches.append(audio_analysis["segments"][i]["pitches"])
            timbres.append(audio_analysis["segments"][i]["timbre"])
            
        return pd.Series([np.array(pitches), np.array(timbres)])