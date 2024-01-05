import json
from typing import Any

import pika
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database

from src.utils.config import config, logger


def save_playlist_counter(db_client: MongoClient, playlist_id: str, num_songs_in_playlist: int) -> None:
    """Saves the counter for processed songs in a playlist to the database.

    Args:
        db_client: The MongoDB client.
        playlist_id: The ID of the playlist.
        num_songs_in_playlist: The total number of songs in the playlist.

    Returns:
        None
    """
    db: Database = db_client.playlist
    collection: Collection = db.processed_songs_counter
    collection.insert_one({"playlist_id": playlist_id, "processed_songs": 0, "total_songs": num_songs_in_playlist})


def save_playlist_info_to_db(db_client: MongoClient, playlist_data: dict[str, str | list[dict[str, Any]]]) -> None:
    """Saves the playlist information to the database.

    Args:
        db_client: The MongoDB client.
        playlist_data: A dictionary containing the playlist data, including the playlist ID and tracks.

    Returns:
        None
    """
    db: Database = db_client.playlist
    collection: Collection = db.playlist_data

    playlist_data["tracks"] = [track.get("id") for track in playlist_data["tracks"]]

    collection.collection.update_one(
        filter={"playlist_id": playlist_data.get("playlist_id")}, update=playlist_data, upsert=True
    )


def send_messages_for_each_song(
    channel: pika.adapters.blocking_connection.BlockingChannel, playlist_data: dict[str, str | list[dict[str, Any]]]
) -> None:
    """Send a message for each song in the playlist to the track message queue.

    Args:
        channel: The RabbitMQ channel for communication.
        playlist_data: A dictionary containing the playlist data, including the tracks.

    Returns:
        None
    """
    queue_name: str = config["tracks_queue"]
    channel.queue_declare(queue=queue_name)
    for track in playlist_data["tracks"]:
        track_dict: str = json.dumps(
            {
                "track": track,
                "playlist_id": playlist_data.get("playlist_id"),
            }
        )
        channel.basic_publish(exchange="", routing_key=queue_name, body=track_dict)


def main(channel: pika.adapters.blocking_connection.BlockingChannel, db_client: MongoClient) -> None:
    """Main function for consuming playlist messages from a message queue.

    Args:
        channel: The RabbitMQ channel.
        db_client: The MongoDB client.

    Returns:
        None
    """
    queue_name: str = config["playlist_queue"]
    channel.queue_declare(queue=queue_name)

    def playlist_unpack(ch, method, properties, body):
        """Unpack the playlist data received from a message queue and performs the following actions.

        - Saves the playlist counter to the database.
        - Saves the playlist information to the database.
        - Sends messages for each song in the playlist to the song's message queue.

        Args:
            ch: The RabbitMQ channel.
            method: The RabbitMQ method.
            properties: The RabbitMQ properties.
            body: The body of the message containing the playlist data.

        Returns:
            None
        """
        body: dict[str, str | int | list[dict[str]]] = json.loads(body)
        save_playlist_counter(db_client, body.get("playlist_id"), body.get("num_songs"))
        save_playlist_info_to_db(db_client, body)
        send_messages_for_each_song(channel, body)

        logger.info(
            f"playlist id: {body.get('playlist_id')} - Total number of songs in the playlist: {body.get('num_songs')}"
        )

    channel.basic_consume(queue=queue_name, on_message_callback=playlist_unpack, auto_ack=True)

    print(" [*] Waiting for messages. To exit press CTRL+C")
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
