"""Contexto opaco de correlación que también se propaga al thread de inferencia."""

from contextvars import ContextVar

request_id_context = ContextVar("ml_request_id", default="startup")
