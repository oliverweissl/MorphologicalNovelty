"""Database schema for CPPNWIN genotype."""

import sqlalchemy
from sqlalchemy.ext.declarative import declarative_base

DbBase = declarative_base()


class FloatDB(DbBase):
    """Stores serialized multineat genomes."""

    __tablename__ = "float"

    id = sqlalchemy.Column(
        sqlalchemy.Integer,
        nullable=False,
        unique=True,
        autoincrement=True,
        primary_key=True,
    )
    value = sqlalchemy.Column(sqlalchemy.Float, nullable=False)
