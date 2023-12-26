import json
import logging
import pathlib
from datetime import datetime
from http import HTTPStatus
from typing import Any

import requests
import yt_dlp
from youtube_search import YoutubeSearch

from src.utils import get_dir_absolute_path, get_global_config, set_logger

with pathlib.Path("config/yt-dlp-config.json").open("r") as fp:
    YOUTUBE_DL_CONFIG: dict[str, str | list[dict[str, str]]] = json.load(fp)


def get_playlist_info(playlist_id: str, access_token: str, offset: int = 0, get_total: bool = True) -> dict[str, Any]:
    """Fetch and return information about a playlist.

    It retrieves the following information:
    - Total number of songs
    - Track
        - Track name (both name and spotify id)
        - Track artist(s) (both name and spotify id)
        - Track album (both name and spotify id)
        - Track release date (both date and date precition)

    Args:
        playlist_id (str): The ID of the playlist.
        access_token (str): The access token for authentication.
        offset (int, optional): The starting offset for fetching tracks. Defaults to 0.
        get_total (bool, optional): Whether to include the total number of tracks in the response. Defaults to True.

    Returns:
        dict[str, Any]: The playlist information as a dictionary.

    Examples:
        >>> playlist_info = get_playlist_info("playlist_id", "access_token")
    """
    logger.info(f"Fetching tracks from playlist id: {playlist_id} starting from the {offset}th song")

    url: str = (
        f"https://api.spotify.com/v1/playlists/{playlist_id}/tracks?market=IT&fields=total,items(track(id, "
        f"uri, name, album(name, release_date, release_date_precision), artists(id, name)))&offset={offset}"
    )
    if not get_total:
        url: str = url.replace("fields=total,", "fields=")

    response: requests.Response = requests.get(url=url, headers={"Authorization": f"Bearer  {access_token}"})
    if response.status_code == HTTPStatus.OK:
        return response.json()

    logger.error(f"Error getting playlist, status code: {response.status_code} - {response.text}")
    return {}


def get_playlist_tracks_data(playlist_id: str, access_token: str) -> list[dict[str, Any]]:
    """Fetch and return the tracks data for a playlist.

    Args:
        playlist_id (str): The ID of the playlist.
        access_token (str): The access token for authentication.

    Returns:
        list[dict[str, Any]]: The tracks data as a list of dictionaries.

    Examples:
        >>> tracks_data = get_playlist_tracks_data("playlist_id", "access_token")
    """
    json_output_dir: pathlib.Path = get_dir_absolute_path("raw")
    playlist_filename: pathlib.Path = json_output_dir / "json" / f"{playlist_id}.json"

    if playlist_filename.exists():
        logger.info("Playlist data already downloaded")
        with pathlib.Path(playlist_filename).open("r") as fp:
            playlist_info: list[dict[str, Any]] = json.load(fp)
        return playlist_info

    offset: int = 0
    num_fetched_songs: int = 0
    total_songs_in_playlist: int = 0
    playlist_info: list[dict[str, Any]] = []

    while True:
        playlist_batch_info: dict[str, Any] = get_playlist_info(
            playlist_id=playlist_id,
            access_token=access_token,
            offset=offset,
            get_total=not bool(total_songs_in_playlist),
        )
        playlist_info.extend(playlist_batch_info.get("items", []))
        num_fetched_songs += len(playlist_batch_info.get("items", []))
        offset += len(playlist_batch_info.get("items", []))

        if total_songs_in_playlist == 0:
            total_songs_in_playlist = playlist_batch_info.get("total", 0)

        if num_fetched_songs >= total_songs_in_playlist:
            break

    return playlist_info


def save_playlist_tracks_data(tracks_data: list[dict[str, Any]], playlist_id: str) -> bool:
    """Save the tracks data for a playlist to a JSON file.

    Args:
        tracks_data (list[dict[str, Any]]): The tracks data to be saved.
        playlist_id (str): The ID of the playlist.

    Returns:
        bool: True if the data was successfully saved, False otherwise.
    """
    json_output_dir: pathlib.Path = get_dir_absolute_path("raw")
    playlist_filename: pathlib.Path = json_output_dir / "json" / f"{playlist_id}.json"

    if playlist_filename.exists():
        return True

    try:
        with pathlib.Path(playlist_filename).open("w") as fp:
            json.dump(tracks_data, fp)
        return True
    except Exception as e:
        logger.error(e)
        return False


def get_song_release_date_folder(release_date: str, precision: str) -> pathlib.Path:
    """Return the folder path based on the song release date.

    Args:
        release_date (str): The release date of the song.
        precision (str): The precision of the release date, can be "year", "month", or "day".

    Returns:
        pathlib.Path: The folder path based on the release date.

    Examples:
        >>> folder_path = get_song_release_date_folder("2022-01-15", "day")
    """
    release_date_precision_map: dict[str, str] = {"year": "%Y", "month": "%Y-%m", "day": "%Y-%m-%d"}
    release_date: datetime = datetime.strptime(release_date, release_date_precision_map[precision]).astimezone()

    return pathlib.Path(release_date.strftime(release_date_precision_map[precision].replace("-", "/")))


def download_song_from_youtube(song_to_search: str, track: dict[str, Any]) -> bool:
    """Download a song from YouTube based on the provided search query and track information.

    Args:
        song_to_search (str): The search query for the song.
        track (dict[str, Any]): The track information.

    Returns:
        bool: True if the song was successfully downloaded, False otherwise.
    """
    release_path: pathlib.Path = get_song_release_date_folder(
        release_date=track.get("track", {})["album"]["release_date"],
        precision=track.get("track", {})["album"]["release_date_precision"],
    )
    output_path: pathlib.Path = get_dir_absolute_path("songs") / release_path / track.get("track", {})["id"]

    # create dirs if they do not exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.with_suffix(".mp3").exists():
        logger.debug(f'Skipping {track.get("track", {}).get("id", "")}. Already downloaded. output_path {output_path}')
        return True
    logger.info(f'To download: {track.get("track", {}).get("id", "")}')
    # return False

    attempts_left: int = 3
    best_url: str = ""
    while attempts_left > 0:
        try:
            url_suffix = YoutubeSearch(song_to_search, max_results=1).to_dict()[0].get("url_suffix")
            best_url = f"https://www.youtube.com{url_suffix}"
            break
        except IndexError:
            attempts_left -= 1
            logger.debug(f"No valid URLs found for {song_to_search}, trying again ({attempts_left} attempts left).")
        if best_url is None:
            logger.debug(f"No valid URLs found for {song_to_search}, skipping track.")
            continue

    # Run you-get to fetch and download the link's audio
    try:
        with yt_dlp.YoutubeDL(YOUTUBE_DL_CONFIG | {"outtmpl": str(output_path)}) as ydl:
            ydl.extract_info(best_url, download=True)
        return True
    except Exception as e:
        logger.error(e)
        return False


def get_string_to_search_in_youtube(track: dict[str, Any]) -> str:
    """Return the string to search for in YouTube based on the track information.

    Args:
        track (dict[str, Any]): The track information.

    Returns:
        str: The string to search for in YouTube.

    Examples:
        >>> search_query = get_string_to_search_in_youtube(track)
    """
    track = track.get("track", {})
    to_search: str = " ".join([artist.get("name") for artist in track.get("artists", [])])
    return f"{to_search} - {track.get('name')}"


config: dict[str, str | float | int | bool] = get_global_config()
logger: logging.Logger = set_logger(config.get("title", ""))
