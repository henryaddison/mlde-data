from enum import Enum


class DomainOption(str, Enum):
    uk = "uk"
    engwales = "engwales"
    scotland = "scotland"


class CollectionOption(str, Enum):
    gcm = "land-gcm"
    cpm = "land-cpm"
    canari_le_sprint = "canari-le-sprint"
