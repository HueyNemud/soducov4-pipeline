from typing import (
    Annotated,
    List,
    Optional,
    Protocol,
    Literal,
    Sequence,
    cast,
)
from typing_extensions import runtime_checkable
from pydantic import BaseModel, Field, RootModel


@runtime_checkable
class ExpandableRootModel(Protocol):
    """Protocol for RootModels that can be expanded to full models."""

    def expand(self) -> BaseModel: ...


# ----------------------------------------------------
# ADRESSE
# ----------------------------------------------------

# Annotations de champs
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
    """Adresse de l'entité extraite."""

    label: AddressLabel
    number: AddressNumber
    complement: AddressAdditional


class AddressCompact(
    RootModel[
        tuple[
            AddressLabel,
            AddressNumber,
            AddressAdditional,
        ]
    ]
):
    """Adresse de l'entité extraite."""

    def expand(self) -> Address:
        label, number, complement = self.root
        return Address(
            label=label,
            number=number,
            complement=complement,
        )


# ----------------------------------------------------
# ENTRÉE D'ANNUAIRE
# ----------------------------------------------------


EntryCat = Annotated[
    Literal["ent"], Field(description="Catégorie de l'entrée d'annuaire.")
]

EntryName = Annotated[
    Optional[str],
    Field(
        description=(
            "Nom de l'entrée : individu, organisation, service. "
            "Optionnellement avec suffixe : titre, complément, détail."
        )
    ),
]

EntryActivity = Annotated[
    Optional[str], Field(description="Activité ou fonction de l'entrée.")
]

EntryAddresses = Annotated[
    Sequence[Address],
    Field(description="Liste des adresses associées."),
]

EntryAdditionalInfo = Annotated[
    List[str],
    Field(
        description=(
            "Toute information additionnelle associée mais non assignable : renvoi, addendum, etc."
        )
    ),
]

EntryLines = Annotated[List[int], Field(description="Index des lignes OCR utilisées.")]


class Entry(BaseModel):
    """Modèle complet pour l'entrée d'annuaire."""

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
    """Modèle compact pour l'entrée d'annuaire."""

    def expand(self) -> Entry:
        (
            cat,
            name,
            activity,
            addresses,
            additional_info,
            lines,
        ) = self.root
        if not all(
            isinstance(addr, ExpandableRootModel) and isinstance(addr, AddressCompact)
            for addr in addresses
        ):
            raise ValueError("All addresses must be AddressCompact instances")
        else:
            addresses = [cast(AddressCompact, addr).expand() for addr in addresses]
        return Entry(
            cat=cat,
            name=name,
            activity=activity,
            addresses=addresses,
            additional_info=additional_info,
            lines=lines,
        )


# ----------------------------------------------------
# TITRE
# ----------------------------------------------------

TitleCat = Annotated[
    Literal["title"], Field(description="Titre de section ou de chapitre.")
]

TitleText = Annotated[str, Field(description="Texte du titre.")]

TitleLines = Annotated[List[int], Field(description="Index des lignes OCR utilisées.")]


class Title(BaseModel):
    """Modèle complet pour le titre."""

    cat: TitleCat
    text: TitleText
    lines: TitleLines


class TitleCompact(
    RootModel[
        tuple[
            TitleCat,
            TitleText,
            TitleLines,
        ]
    ]
):
    """Modèle compact pour le titre."""

    def expand(self) -> Title:
        cat, text, lines = self.root
        return Title(
            cat=cat,
            text=text,
            lines=lines,
        )


# ----------------------------------------------------
# TEXTE LIBRE
# ----------------------------------------------------


TextCat = Annotated[Literal["txt"], Field(description="Texte isolé.")]

TextText = Annotated[str, Field(description="Contenu du texte.")]

TextLines = Annotated[List[int], Field(description="Index de la ligne OCR utilisée.")]


class Text(BaseModel):
    """Modèle complet pour le texte libre."""

    cat: TextCat
    text: TextText
    lines: TextLines


class TextCompact(
    RootModel[
        tuple[
            TextCat,
            TextText,
            TextLines,
        ]
    ]
):
    """Modèle compact pour le texte libre."""

    def expand(self) -> Text:
        cat, text, lines = self.root
        return Text(
            cat=cat,
            text=text,
            lines=lines,
        )


# ----------------------------------------------------
# STRUCTURE GENERALE
# ----------------------------------------------------


# -----------------------------
# Union et Structure globale
# -----------------------------

StructuredItem = Entry | Title | Text
StructuredItemCompact = EntryCompact | TitleCompact | TextCompact


class Structured(BaseModel):
    items: list[StructuredItem] = Field(..., description="Liste d'items extraits.")

    def __iter__(self):
        return iter(self.items)

    def __getitem__(self, item):
        return self.items[item]

    def __len__(self):
        return len(self.items)


class StructuredCompact(
    RootModel[
        tuple[
            StructuredItemCompact,
            ...,
        ]
    ]
):
    """Modèle compact pour la structure globale."""

    def expand(self) -> Structured:
        expanded_items = []
        for item in self.root:
            if isinstance(item, ExpandableRootModel):
                expanded_items.append(item.expand())
            else:
                raise ValueError("Item does not implement ExpandableRootModel")
        return Structured(items=expanded_items)


class StructuredSeq(RootModel[Sequence[Structured | StructuredCompact]]):
    """Modèle pour une séquence d'items structurés."""

    def __iter__(self):
        return iter(self.root)

    def __getitem__(self, item):
        return self.root[item]

    def __len__(self):
        return len(self.root)
