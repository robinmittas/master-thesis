from typing import NamedTuple


class _REGISTRY_KEYS_NUCLEUS_CYTOSOL_MODEL(NamedTuple):
    """Naming Convention.

    Define the namings of respective abundances (needs to align with preprocessed data)
    U_NUC = Unspliced abundance in nucleus
    S_NUC = Spliced abundance in nucleus
    S_CYT = Spliced abundance in cytoplasm.
    """

    U_NUC_KEY = "U_NUC"
    S_NUC_KEY = "X"
    S_CYT_KEY = "S_CYT"


REGISTRY_KEYS = _REGISTRY_KEYS_NUCLEUS_CYTOSOL_MODEL()
