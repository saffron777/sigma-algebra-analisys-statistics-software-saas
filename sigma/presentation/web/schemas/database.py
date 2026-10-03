"""Request model for choosing the database (the fields are checked again, strictly, by ``DatabaseSettings``)."""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

from sigma.presentation.web.schemas.base import RequestModel

Short = Annotated[str, Field(max_length=256)]


class DatabaseRequest(RequestModel):
    engine: Annotated[str, Field(min_length=1, max_length=16)]
    host: Short | None = None
    port: int | None = Field(default=None, ge=1, le=65535)
    database: Short | None = None
    schema_name: Short | None = Field(default=None, alias="schema")
    username: Short | None = None
    password: Annotated[str, Field(max_length=256)] | None = None
    options: dict[str, str] = Field(default_factory=dict)
    migrate: bool = False

    model_config = {**RequestModel.model_config, "populate_by_name": True}

    def payload(self) -> dict:
        return {
            "engine": self.engine,
            "host": self.host,
            "port": self.port,
            "database": self.database,
            "schema": self.schema_name,
            "username": self.username,
            "password": self.password,
            "options": self.options,
            "migrate": self.migrate,
        }
