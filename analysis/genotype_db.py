"""Database schema for CPPNWIN genotype."""

import sqlalchemy
from sqlalchemy.ext.declarative import declarative_base

DbBase = declarative_base()


class GenotypeDB(DbBase):
    """Stores serialized multineat genomes."""

    __tablename__ = "genotype"

    id = sqlalchemy.Column(
        sqlalchemy.Integer,
        nullable=False,
        unique=True,
        autoincrement=True,
        primary_key=True,
    )
    body_id = sqlalchemy.Column(sqlalchemy.Integer, nullable=False)
    brain_id = sqlalchemy.Column(sqlalchemy.Integer, nullable=False)