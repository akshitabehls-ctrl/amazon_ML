"""Road and unit abbreviation canonical maps (English + French)."""

ROAD_ABBREV_MAP = {
    "rd": "road", "road": "road",
    "st": "street", "street": "street",
    "ave": "avenue", "av": "avenue", "avenue": "avenue",
    "blvd": "boulevard", "boulevard": "boulevard", "bd": "boulevard",
    "dr": "drive", "drive": "drive",
    "cir": "circle", "circle": "circle",
    "ter": "terrace", "terrace": "terrace",
    "ln": "lane", "lane": "lane",
    "rue": "rue",  # French, no abbreviation
}

UNIT_TOKENS = frozenset({"unit", "apt", "suite", "ste", "pmb", "fl", "floor", "#"})
