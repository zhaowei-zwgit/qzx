"""SAM2-UNet model variants with lazy public exports."""

from importlib import import_module


_EXPORTS = {
    "BaselineSAM2UNet": (".baseline", "SAM2UNet"),
    "ExperimentalDarkIRSAM2UNet": (".experimental_darkir", "SAM2UNet"),
    "SAM2UNetFusion": (".fusion", "SAM2UNetFusion"),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from error
    return getattr(import_module(module_name, __name__), attribute_name)
