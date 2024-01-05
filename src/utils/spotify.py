import pathlib
from datetime import datetime
from http import HTTPStatus
from typing import Any

import requests
import yt_dlp
from cachetools.func import ttl_cache
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from youtube_search import YoutubeSearch

from .config import logger
from .utils import get_dir_absolute_path

youtube_dl_options: dict[str, str | list[dict[str, str]]] = {
    "quiet": True,
    "format": "bestaudio/best",
    "embedthumbnail": True,
    "postprocessors": [
        {
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        },
        {
            "key": "FFmpegMetadata",
        },
    ],
}


def get_song_realease_date_folder(release_date: str, precision: str) -> pathlib.Path:
    release_date_precision_map: dict[str, str] = {"year": "%Y", "month": "%Y-%m", "day": "%Y-%m-%d"}
    release_date: datetime = datetime.strptime(release_date, release_date_precision_map[precision])

    return pathlib.Path(release_date.strftime(release_date_precision_map[precision].replace("-", "/")))


@ttl_cache(maxsize=1, ttl=3600)
def get_spotify_access_token(client_id: str, client_secret: str) -> str:
    """Return the Spotify access token.

    Returns:
        str: The Spotify access token.

    Examples:
        >>> access_token = get_spotify_access_token()
    """
    response: requests.Response = requests.post(
        url="https://accounts.spotify.com/api/token",
        data=f"grant_type=client_credentials&client_id={client_id}&client_secret={client_secret}",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    return response.json().get("access_token") if response.status_code == HTTPStatus.OK else ""


def download_song_from_youtube(song_to_search: str, track: dict[str, Any]) -> pathlib.Path | None:
    release_path: pathlib.Path = get_song_realease_date_folder(
        release_date=track["track"]["album"]["release_date"],
        precision=track["track"]["album"]["release_date_precision"],
    )
    output_path: pathlib.Path = get_dir_absolute_path("songs") / release_path / track["track"]["id"]

    # create dirs if they do not exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.with_suffix(".mp3").exists():
        logger.debug(f"Skipping {track['track']['id']}. Already downloaded. output_path {output_path}")
        return True
    logger.info(f"To download: {track['track']['id']}")
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
    # print(f"Initiating download for {song_to_search}. url is: {best_url}")
    try:
        with yt_dlp.YoutubeDL(youtube_dl_options | {"outtmpl": str(output_path)}) as ydl:
            ydl.extract_info(best_url, download=True)
        return output_path.with_suffix(".mp3")
    except Exception as e:
        logger.error(e)
        return None


def get_string_to_search_in_youtube(track: dict[str, Any]) -> str:
    track: dict[str, dict[str, str | list[str]] | str] = track.get("track", {})
    to_search: str = " ".join([artist.get("name") for artist in track["artists"]])
    return f"{to_search} - {track.get('name')}"


def download_song(track: dict[str, dict[str, Any]]) -> pathlib.Path | None:
    to_search: str = get_string_to_search_in_youtube(track)
    return download_song_from_youtube(to_search, track)


def get_playlist_info(playlist_id: str, access_token: str, offset: int = 0, get_total: bool = True) -> dict[str, Any]:
    logger.info(f"Fetching tracks from playlist id: {playlist_id} starting from the {offset}th song")
    url: str = f"https://api.spotify.com/v1/playlists/{playlist_id}/tracks?market=IT&fields=total,items(track(id, uri, name, album(name, release_date, release_date_precision), artists(id, name)))&offset={offset}"
    if not get_total:
        url: str = url.replace("fields=total,", "fields=")

    response: requests.Response = requests.get(url=url, headers={"Authorization": f"Bearer  {access_token}"})
    if response.status_code == HTTPStatus.OK:
        return response.json()

    print(f"Error getting playlist, status code: {response.status_code} - {response.text}")
    return {}


def get_playlist_tracks_data(playlist_id: str, access_token: str) -> dict[str, str | list[dict[str, Any]]]:
    offset: int = 0
    num_fetched_songs: int = 0
    total_songs_in_playlist: int = 0

    playlist_info: list[dict[str, Any]] = []

    logger.info(
        f"Fetching playlist tracks for {playlist_id}, current offset: {offset}, total songs: {total_songs_in_playlist}, fetching total {not bool(total_songs_in_playlist)}, fetched {num_fetched_songs} songs"
    )
    while True:
        playlist_batch_info: dict[str, Any] = get_playlist_info(
            playlist_id, access_token, offset=offset, get_total=not bool(total_songs_in_playlist)
        )
        playlist_info.extend(playlist_batch_info.get("items", []))

        if total_songs_in_playlist == 0:
            total_songs_in_playlist = playlist_batch_info.get("total", 0)
        num_fetched_songs += len(playlist_batch_info.get("items", []))
        offset += len(playlist_batch_info.get("items", []))
        logger.info(
            f"Fetched {num_fetched_songs}, total {total_songs_in_playlist}, offset {offset}, length {len(playlist_batch_info.get('items', []))}"
        )
        if num_fetched_songs >= total_songs_in_playlist:
            break

    return {"playlist_id": playlist_id, "num_songs": total_songs_in_playlist, "tracks": playlist_info}


def was_playlist_updated(playlist_data: dict[str, str | list[dict[str, Any]]], db_playlist: dict[str, str]) -> bool:
    new_playlist_tracks: set[str] = set(playlist_data.get("tracks", []))
    old_playlist_tracks: set[str] = set(db_playlist.get("tracks", []))

    new_songs: list[str] = list(new_playlist_tracks.difference(old_playlist_tracks))
    deleted_songs: list[str] = list(old_playlist_tracks.difference(new_playlist_tracks))
    return bool(new_songs) or bool(deleted_songs)


def save_playlist_info_to_db(playlist_data: dict[str, str | list[dict[str, Any]]]) -> None:
    client: MongoClient = MongoClient(
        "mongodb://127.0.0.1:27017/?directConnection=true&serverSelectionTimeoutMS=2000&appName=mongosh+2.1.1"
    )
    db = client.playlist
    collection = db.playlist_data

    playlist_data["tracks"] = [track.get("id") for track in playlist_data["tracks"]]

    # result: dict[str, str] = collection.find_one({"playlist_id": playlist_data.get("playlist_id")})
    collection.collection.update_one(
        filter={"playlist_id": playlist_data.get("playlist_id")}, update=playlist_data, upsert=True
    )
    #
    # if not result:
    #     collection.insert_one(playlist_data)
    #     return
    #
    # if was_playlist_updated(playlist_data, result):
    #     collection.collection.update_one(
    #         {"playlist_id": playlist_data.get("playlist_id")}, {"$set": {"tracks": playlist_data.get("tracks")}}
    #     )


def save_playlist_counter(playlist_id: str, num_songs_in_playlist: int) -> None:
    client: MongoClient = MongoClient(
        "mongodb://127.0.0.1:27017/?directConnection=true&serverSelectionTimeoutMS=2000&appName=mongosh+2.1.1"
    )
    db = client.playlist
    collection = db.processed_songs_counter
    collection.insert_one({"playlist_id": playlist_id, "processed_songs": 0, "total_songs": num_songs_in_playlist})


def delete_playlist_counter(collection: Collection, playlist_id: str) -> None:
    collection.delete_one(filter={"playlist_id": playlist_id})


def send_playlist_processed_event():
    # roba con rabbitmq
    return


def update_playlist_counter(playlist_id: str) -> None:
    client: MongoClient = MongoClient(
        "mongodb://127.0.0.1:27017/?directConnection=true&serverSelectionTimeoutMS=2000&appName=mongosh+2.1.1"
    )
    db: Database = client.playlist
    collection: Collection = db.processed_songs_counter
    collection.update_one({"playlist_id": playlist_id}, {"$inc": {"processed_songs": 1}})
    updated_counter: dict[str, str | int] = collection.find_one(
        {"playlist_id": playlist_id}, projection={"processed_songs": 1, "total_songs": 1}
    )
    if updated_counter.get("processed_songs") == updated_counter.get("total_songs"):
        send_playlist_processed_event(playlist_id)
        delete_playlist_counter(collection, playlist_id)
