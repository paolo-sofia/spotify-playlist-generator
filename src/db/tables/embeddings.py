from numbers import Number

from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, Index, String
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class SongEmbedding(Base):
    __tablename__ = "song_embedding"
    __table_args__ = (
        Index("idx_id", "id"),
        Index(
            "hnsw_core_index",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    id: str = Column(String, primary_key=True)
    embedding: list[Number] = Column(Vector(256))
