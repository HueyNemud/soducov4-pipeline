"""
Structured Data Extraction Schemas.

This module defines the models for directory listings, including addresses,
entries, titles, and free text. It provides both full and compact
representations (RootModels) to optimize JSON payloads.
"""

from typing import (
    Annotated,
    List,
    Optional,
    Protocol,
    Literal,
    Sequence,
    Union,
)
from typing_extensions import runtime_checkable
from pydantic import BaseModel, Field, RootModel


@runtime_checkable
class ExpandableRootModel(Protocol):
    """Protocol for compact models that can be expanded into full Pydantic models."""

    def expand(self) -> BaseModel: ...


# =============================================================================
# ADDRESS MODELS
# =============================================================================

AddressLabel = Annotated[
    Optional[str], Field(description="Libellé de l'adresse: voie ou lieu")
]
AddressNumber = Annotated[
    Optional[str], Field(description="Numéro si disponible. Situé après le label.")
]
AddressAdditional = Annotated[
    Optional[str],
    Field(description="Informations complémentaires : quartier, bâtiment, etc."),
]


class Address(BaseModel):
    """Full representation of a geographical address."""

    label: AddressLabel
    number: AddressNumber
    complement: AddressAdditional


class AddressCompact(RootModel[tuple[AddressLabel, AddressNumber, AddressAdditional]]):
    """Compact tuple-based representation of an Address."""

    def expand(self) -> Address:
        """Transforms the tuple back into a structured Address object."""
        label, number, complement = self.root
        return Address(label=label, number=number, complement=complement)


# =============================================================================
# DIRECTORY ENTRY MODELS
# =============================================================================

EntryCat = Annotated[
    Literal["ent"], Field(description="Catégorie de l'entrée d'annuaire.")
]
EntryName = Annotated[
    Optional[str],
    Field(
        description="Nom de l'entrée : individu, organisation, service. Optionnellement avec suffixe."
    ),
]
EntryActivity = Annotated[
    Optional[str], Field(description="Activité ou fonction de l'entrée.")
]
EntryAddresses = Annotated[
    Sequence[Address], Field(description="Liste des adresses associées.")
]
EntryAdditionalInfo = Annotated[
    List[str],
    Field(description="Toute information additionnelle associée mais non assignable."),
]
EntryLines = Annotated[List[int], Field(description="Index des lignes OCR utilisées.")]


class Entry(BaseModel):
    """Full representation of a directory listing entry."""

    cat: EntryCat
    name: EntryName
    activity: EntryActivity
    addresses: EntryAddresses
    additional_info: EntryAdditionalInfo
    lines: EntryLines


class EntryCompact(
    RootModel[
        tuple[
            EntryCat,
            EntryName,
            EntryActivity,
            EntryAddresses,
            EntryAdditionalInfo,
            EntryLines,
        ]
    ]
):
    """Compact tuple-based representation of a Directory Entry."""

    def expand(self) -> Entry:
        """Expands the compact entry and its nested compact addresses into full models."""
        cat, name, activity, addresses, info, lines = self.root

        # Ensure nested addresses are expanded if they are in compact form
        expanded_addresses = [
            addr.expand() if isinstance(addr, ExpandableRootModel) else addr
            for addr in addresses
        ]

        validated_addresses: EntryAddresses = [
            Address.model_validate(a) for a in expanded_addresses
        ]

        return Entry(
            cat=cat,
            name=name,
            activity=activity,
            addresses=validated_addresses,
            additional_info=info,
            lines=lines,
        )


# =============================================================================
# TITLE & FREE TEXT MODELS
# =============================================================================

TitleCat = Annotated[
    Literal["title"], Field(description="Titre de section ou de chapitre.")
]
TitleText = Annotated[str, Field(description="Texte du titre.")]
TitleLines = Annotated[List[int], Field(description="Index des lignes OCR utilisées.")]


class Title(BaseModel):
    """Full representation of a section title."""

    cat: TitleCat
    text: TitleText
    lines: TitleLines


class TitleCompact(RootModel[tuple[TitleCat, TitleText, TitleLines]]):
    """Compact representation of a Title."""

    def expand(self) -> Title:
        cat, text, lines = self.root
        return Title(cat=cat, text=text, lines=lines)


TextCat = Annotated[Literal["txt"], Field(description="Texte isolé.")]
TextText = Annotated[str, Field(description="Contenu du texte.")]
TextLines = Annotated[List[int], Field(description="Index de la ligne OCR utilisée.")]


class Text(BaseModel):
    """Full representation of a free text block."""

    cat: TextCat
    text: TextText
    lines: TextLines


class TextCompact(RootModel[tuple[TextCat, TextText, TextLines]]):
    """Compact representation of Free Text."""

    def expand(self) -> Text:
        cat, text, lines = self.root
        return Text(cat=cat, text=text, lines=lines)


# =============================================================================
# GLOBAL STRUCTURE
# =============================================================================

StructuredItem = Union[Entry, Title, Text]
StructuredItemCompact = Union[EntryCompact, TitleCompact, TextCompact]


class Structured(BaseModel):
    """Container for a collection of extracted items."""

    items: List[StructuredItem] = Field(..., description="Liste d'items extraits.")

    def __iter__(self):
        return iter(self.items)

    def __getitem__(self, index):
        return self.items[index]

    def __len__(self):
        return len(self.items)


class StructuredCompact(RootModel[List[StructuredItemCompact]]):
    """Global collection of items in compact format."""

    def expand(self) -> Structured:
        """Iterates through compact items and expands each one into its full model."""
        return Structured(
            items=[
                item.expand()
                for item in self.root
                if isinstance(item, ExpandableRootModel)
            ]
        )


class StructuredSeq(RootModel[Sequence[Union[Structured, StructuredCompact]]]):
    """A flexible sequence containing either full or compact structured data."""

    def __iter__(self):
        return iter(self.root)

    def __getitem__(self, index):
        return self.root[index]

    def __len__(self):
        return len(self.root)
