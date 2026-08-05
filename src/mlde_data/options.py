from enum import Enum


class DomainOption(str, Enum):
    uk = "uk"
    engwales = "engwales"
    scotland = "scotland"
    englandwales_5km = "engwales-5km"
    scotland_5km = "scotland-5km"


class CollectionOption(str, Enum):
    gcm = "land-gcm"
    cpm = "land-cpm"
    canari_le_sprint = "canari-le-sprint"
