"""Updater role boundary."""

from face_of_agi.models.updater.contracts import UpdaterTaskRegistry

Updater = UpdaterTaskRegistry

__all__ = ["Updater", "UpdaterTaskRegistry"]
