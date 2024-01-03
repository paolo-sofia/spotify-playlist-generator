from pydantic import BaseModel


class SongEmbedding(BaseModel):
    id: str
    embedding: list[float | int]


class SongEmbeddingCreate(SongEmbedding):
    pass


class SongEmbeddingUpdate(SongEmbedding):
    pass


class SongEmbeddingOrm(SongEmbedding):
    class Config:
        orm_mode = True
