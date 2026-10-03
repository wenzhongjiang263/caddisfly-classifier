"""Second-generation training and inference pipeline for caddisfly images."""

from .labels import SEX_STATUSES, parse_folder_label

__all__ = ["SEX_STATUSES", "parse_folder_label"]
