"""Canonical label parsing shared by data preparation, training, and evaluation."""

from __future__ import annotations

from dataclasses import dataclass


SEX_STATUSES = ("female", "male", "mating")

# The source collections contain a few historical spellings and infraspecific
# annotations.  Keep this mapping in one visible place so evaluation is stable.
SPECIES_ALIASES = {
    "Asmicridea edwardsi": "Asmicridea edwardsii",
    "Asmicridea edwardsii (mottled)": "Asmicridea edwardsii",
    "Austrheithus glymma": "Austrheithrus glymma",
    "Tamasia palpata": "Tasimia palpata",
    "Triplectides ciuskus ciuskus": "Triplectides ciuskus",
    "Triplectides similis (large eye)": "Triplectides similis",
}


@dataclass(frozen=True)
class ParsedLabel:
    family: str
    species: str
    sex_status: str
    morph_status: str


def canonical_species(value: str) -> str:
    """Return a stripped, canonical species label."""

    cleaned = " ".join(value.strip().split())
    return SPECIES_ALIASES.get(cleaned, cleaned)


def parse_folder_label(folder_name: str) -> ParsedLabel:
    """Parse ``Family Genus species sex`` directory names.

    The historical Leptoceridae folders contain a trailing comma after the
    family name; it is deliberately normalized here.
    """

    cleaned = " ".join(folder_name.strip().split())
    try:
        body, sex_status = cleaned.rsplit(" ", 1)
    except ValueError as exc:
        raise ValueError(f"Cannot parse class directory: {folder_name!r}") from exc
    if sex_status not in SEX_STATUSES:
        raise ValueError(
            f"Class directory must end in one of {SEX_STATUSES}: {folder_name!r}"
        )
    try:
        family, species = body.split(" ", 1)
    except ValueError as exc:
        raise ValueError(f"Missing species in class directory: {folder_name!r}") from exc
    family = family.rstrip(",")
    morph_status = ""
    if species.endswith(" (mottled)"):
        species = species.removesuffix(" (mottled)")
        morph_status = "mottled"
    elif species == "Asmicridea edwardsii":
        morph_status = "typical"
    return ParsedLabel(
        family=family,
        species=canonical_species(species),
        sex_status=sex_status,
        morph_status=morph_status,
    )
