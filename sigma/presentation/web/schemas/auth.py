"""Request models for accounts and authentication."""

from __future__ import annotations

from typing import Literal

from sigma.presentation.web.schemas.base import NonEmpty, RequestModel


class LoginRequest(RequestModel):
    username: NonEmpty
    password: NonEmpty


class SetupRequest(RequestModel):
    username: NonEmpty
    display_name: str | None = None
    password: NonEmpty


class ChangePasswordRequest(RequestModel):
    current_password: NonEmpty
    new_password: NonEmpty


class ProfileRequest(RequestModel):
    display_name: NonEmpty


class CreateUserRequest(RequestModel):
    username: NonEmpty
    display_name: str | None = None
    password: NonEmpty
    role: Literal["admin", "user"] = "user"
    can_use_console: bool = False
    can_use_ai: bool = False


class UpdateUserRequest(RequestModel):
    display_name: str | None = None
    role: Literal["admin", "user"] | None = None
    is_active: bool | None = None
    can_use_console: bool | None = None
    can_use_ai: bool | None = None


class ResetPasswordRequest(RequestModel):
    new_password: NonEmpty
