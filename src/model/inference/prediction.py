import json
from typing import Iterator

import pika
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database

from src.db.queries.embeddings import insert_embeddings
from src.utils.config import config, logger


def generate_playlist(db_client: MongoClient, playlist_id: str) -> bool:
    tracks: list[dict[str, str]] = get_playlist_tracks(db_client, playlist_id)
    embeddings: list[dict[str, str | list[float]]] = [get_song_embedding(track) for track in tracks]
    closest_embeddings: list[dict[str, str | list[float]]] = get_closest_embeddings(embeddings, config["model"]["k"])
    insert_embeddings(embeddings)
    extract_playlist_from_embeddings(embeddings, closest_embeddings)
    publish_playlist()
    return True


def get_playlist_tracks(db_client: MongoClient, playlist_id: str) -> list[dict[str, str]]:
    db: Database = db_client[config["mongodb"]["db_name"]]
    collection: Collection = db[config["mongodb"]["song_collection"]]
    tracks: Iterator = collection.find({"playlist_id": playlist_id}, projection={"song_id": 1, "audio_path": 1})
    return list(tracks)


def main(channel: pika.adapters.blocking_connection.BlockingChannel, db_client: MongoClient) -> None:
    queue_name: str = config["playlist_queue"]
    channel.queue_declare(queue=queue_name)

    def playlist_callback(ch, method, properties, body):
        playlist_id: dict[str, str] = json.loads(body).get("playlist_id")
        logger.info(f"{playlist_id} download finished. Starting prediction process")
        generate_playlist(db_client, playlist_id)

    channel.basic_consume(queue=queue_name, on_message_callback=playlist_callback, auto_ack=True)
    channel.start_consuming()


if __name__ == "__main__":
    connection: pika.BlockingConnection = pika.BlockingConnection(
        pika.ConnectionParameters(host=config["rabbitmq"]["host"], port=config["rabbitmq"]["port"])
    )
    channel: pika.adapters.blocking_connection.BlockingChannel = connection.channel()

    client: MongoClient = MongoClient(
        f"mongodb://{config['mongodb']['host']}:{config['mongodb']['port']}/?directConnection=true"
        f"&serverSelectionTimeoutMS=2000&appName=mongosh+2.1.1"
    )
    main(channel, client)
    connection.close()
