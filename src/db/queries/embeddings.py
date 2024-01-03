from sqlalchemy import select
from sqlalchemy.orm import Session

from src.db.schemas.song_embedding import SongEmbedding
from src.db.tables.embeddings import SongEmbedding as SongEmbeddingSQL


def add_object(db: Session, obj: SongEmbeddingSQL) -> SongEmbeddingSQL:
    db.add(obj)
    db.commit()
    db.refresh(obj)
    return obj


def insert_embeddings(db: Session, songs_embedding: SongEmbedding | list[SongEmbedding]) -> None:
    if not isinstance(songs_embedding, (list, SongEmbedding)):
        print("songs_embedding is not of type SongEmbedding or list[SongEmbedding]")
        return

    if isinstance(songs_embedding, SongEmbedding):
        add_object(db, songs_embedding)
        return

    for song_embedding in songs_embedding:
        add_object(db, song_embedding)
    return


def get_embeddings(db: Session, songs_id: str | list[str]) -> SongEmbeddingSQL | list[SongEmbeddingSQL]:
    if not isinstance(songs_id, (list, str)):
        print("songs_embedding is not of type SongEmbedding or list[SongEmbedding]")
        return []

    if isinstance(songs_id, str):
        return db.query(SongEmbeddingSQL).filter(SongEmbedding.id == songs_id).first()

    return db.query(SongEmbeddingSQL).filter(SongEmbedding.id.in_(songs_id))


def update_single_embedding(db: Session, songs_embedding: SongEmbedding | list[SongEmbedding]) -> None:
    if not isinstance(songs_embedding, SongEmbedding):
        print("songs_embedding is not of type SongEmbedding")

    db_song: SongEmbedding = get_embeddings(db, songs_embedding.id)
    if not db_song:
        print(f"Song {songs_embedding.id} not present in the db")
        return
    try:
        db.query(SongEmbeddingSQL).filter(SongEmbedding.id == songs_embedding.id).update(songs_embedding)
    except Exception as e:
        print(e)

    return


def update_embeddings(db: Session, songs_embedding: list[SongEmbedding]) -> None:
    if not isinstance(songs_embedding, (list, SongEmbedding)):
        print("songs_embedding is not of type SongEmbedding or list[SongEmbedding]")
        return None

    if isinstance(songs_embedding, SongEmbedding):
        return update_single_embedding(db, SongEmbeddingSQL(id=songs_embedding.id, embedding=songs_embedding.embedding))

    for song_embedding in songs_embedding:
        update_single_embedding(db, SongEmbeddingSQL(id=song_embedding.id, embedding=song_embedding.embedding))

    return None


def get_closest_embedding(
    db: Session, songs_embedding: SongEmbedding | list[SongEmbedding], k: int = 3
) -> list[SongEmbeddingSQL]:
    if not isinstance(songs_embedding, (list, SongEmbedding)):
        print("songs_embedding is not of type SongEmbedding or list[SongEmbedding]")
        return []

    if isinstance(songs_embedding, SongEmbedding):
        return db.query(SongEmbeddingSQL).order_by(SongEmbedding.embedding.cosine_distance(songs_embedding.embedding)).limit(k).all()


    nearest_neighbors_songs: list[SongEmbedding] = []
    for song_embedding in songs_embedding:
        nearest_neighbors_songs.extend(
            db.query(SongEmbeddingSQL).order_by(SongEmbedding.embedding.cosine_distance(song_embedding.embedding)).limit(k).all()
        )

    return nearest_neighbors_songs
