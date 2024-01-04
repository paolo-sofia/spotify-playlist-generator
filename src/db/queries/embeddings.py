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
        print("insert_embeddings - songs_embedding is not of type SongEmbedding or list[SongEmbedding]")
        return

    if isinstance(songs_embedding, SongEmbedding):
        add_object(db, SongEmbeddingSQL(**songs_embedding.model_dump()))
        return

    for song_embedding in songs_embedding:
        add_object(db, SongEmbeddingSQL(**song_embedding.model_dump()))
    return


def get_embeddings(
    db: Session, songs_id: str | list[str] | SongEmbedding | None = None
) -> None | SongEmbeddingSQL | list[SongEmbeddingSQL]:
    if not songs_id:
        return db.query(SongEmbeddingSQL).all()

    if not isinstance(songs_id, (list, str, SongEmbedding)):
        print("get_embeddings - songs_embedding is not of type SongEmbedding or list[SongEmbedding]")
        return None

    if isinstance(songs_id, str):
        return db.query(SongEmbeddingSQL).filter(SongEmbeddingSQL.id == songs_id).first()
    if isinstance(songs_id, SongEmbedding):
        return db.query(SongEmbeddingSQL).filter(SongEmbeddingSQL.id == songs_id.id).first()

    return db.query(SongEmbeddingSQL).filter(SongEmbeddingSQL.id.in_(songs_id)).all()


def _update_single_embedding(db: Session, song_embedding: SongEmbedding) -> None:
    if not isinstance(song_embedding, SongEmbedding):
        print("update_single_embedding - songs_embedding is not of type SongEmbedding")

    db_song: SongEmbedding = get_embeddings(db, song_embedding.id)
    if not db_song:
        print(f"Song {song_embedding.id} not present in the db")
        return
    try:
        db.query(SongEmbeddingSQL).filter(SongEmbeddingSQL.id == song_embedding.id).update(
            {SongEmbeddingSQL.embedding.name: song_embedding.embedding}
        )
    except Exception as e:
        print(e)


def update_embeddings(db: Session, songs_embedding: SongEmbedding | list[SongEmbedding]) -> None:
    if not isinstance(songs_embedding, (list, SongEmbedding)):
        print("update_embeddings - songs_embedding is not of type SongEmbedding or list[SongEmbedding]")
        return None

    if isinstance(songs_embedding, SongEmbedding):
        return _update_single_embedding(db, songs_embedding)

    for song_embedding in songs_embedding:
        _update_single_embedding(db, song_embedding)

    return None


def get_closest_embedding(
    db: Session, songs_embedding: SongEmbedding | list[SongEmbedding], k: int = 3
) -> set[SongEmbeddingSQL]:
    if not isinstance(songs_embedding, (list, SongEmbedding)):
        print("songs_embedding is not of type SongEmbedding or list[SongEmbedding]")
        return set()

    if isinstance(songs_embedding, SongEmbedding):
        return set(
            db.query(SongEmbeddingSQL)
            .order_by(SongEmbeddingSQL.embedding.cosine_distance(songs_embedding.embedding))
            .limit(k)
            .all()
        )

    nearest_neighbors_songs: list[SongEmbedding] = []
    for song_embedding in songs_embedding:
        nearest_neighbors_songs.extend(
            db.query(SongEmbeddingSQL)
            .order_by(SongEmbeddingSQL.embedding.cosine_distance(song_embedding.embedding))
            .limit(k)
            .all()
        )

    return set(nearest_neighbors_songs)
