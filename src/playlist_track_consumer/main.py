import json
import pathlib
from typing import Any

import pika
from pika.adapters.blocking_connection import BlockingChannel
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database

from src.utils.config import config
from src.utils.spotify import download_song


def delete_playlist_counter(collection: Collection, playlist_id: str) -> None:
    collection.delete_one(filter={"playlist_id": playlist_id})


def send_playlist_processed_event(connection: pika.BlockingConnection, playlist_id: str) -> None:
    channel: BlockingChannel = connection.channel()
    channel.queue_declare(config["rabbitmq"]["counter"])
    channel.basic_publish(
        exchange="", routing_key=config["rabbitmq"]["counter"], body=json.dumps({"playlist_id": playlist_id})
    )


def update_playlist_counter(db_client: MongoClient, connection: pika.BlockingConnection, playlist_id: str) -> None:
    db: Database = db_client.playlist
    collection: Collection = db.processed_songs_counter

    collection.update_one({"playlist_id": playlist_id}, {"$inc": {"processed_songs": 1}})
    updated_counter: dict[str, str | int] = collection.find_one(
        {"playlist_id": playlist_id}, projection={"processed_songs": 1, "total_songs": 1}
    )
    if updated_counter.get("processed_songs") != updated_counter.get("total_songs"):
        return

    send_playlist_processed_event(connection, playlist_id)
    delete_playlist_counter(collection, playlist_id)


def update_song_with_audio_path(
    db_client: MongoClient, song: dict[str, str | int | list[dict[str]]], song_audio_path: pathlib.Path
) -> None:
    db: Database = db_client[config["mongodb"]["database"]]
    collection: Collection = db[config["mongodb"]["song_collection"]]
    collection.update_one(filter={"song_id": song["song_id"]}, update={"$set": {"audio_path": str(song_audio_path)}})


def main(connection: pika.BlockingConnection, db_client: MongoClient) -> None:
    queue_name: str = config["song_queue"]

    channel: pika.adapters.blocking_connection.BlockingChannel = connection.channel()
    channel.queue_declare(queue=queue_name)

    def song_callback(ch, method, properties, body):
        message: dict[str, str | dict[str, dict[str, Any]]] = json.loads(body)
        output_path: pathlib.Path | None = download_song(message["track"])
        if output_path:
            update_song_with_audio_path(db_client, message, output_path)
        update_playlist_counter(db_client, connection, message["playlist_id"])

    channel.basic_consume(queue=queue_name, on_message_callback=song_callback, auto_ack=True)
    channel.start_consuming()


if __name__ == "__main__":
    connection: pika.BlockingConnection = pika.BlockingConnection(
        pika.ConnectionParameters(host=config["rabbitmq"]["host"], port=config["rabbitmq"]["port"])
    )

    client: MongoClient = MongoClient(
        f"mongodb://{config['mongodb']['host']}:{config['mongodb']['port']}/?directConnection=true"
        f"&serverSelectionTimeoutMS=2000&appName=mongosh+2.1.1"
    )
    main(connection, client)
    connection.close()
