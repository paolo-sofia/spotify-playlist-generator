import dataclasses
import pathlib


def get_dir_absolute_path(dir_name: str) -> pathlib.Path:
    """Return the absolute path of the directory with the specified name.

    It searches in all subdirectories of the cwd parent folder and returns the absolute path of the directory named
    `dir_name`
    Args:
        dir_name (str): The name of the directory.

    Returns:
        pathlib.Path: The absolute path of the directory.

    Examples:
        >>> dir_path = get_dir_absolute_path("my_directory")
    """
    current_folder: pathlib.Path = pathlib.Path.cwd()

    target_folder_path: pathlib.Path = pathlib.Path()
    for parent in current_folder.parents:
        for potential_folder_path in parent.rglob(dir_name):
            if potential_folder_path.is_dir():
                return potential_folder_path

    return target_folder_path


def dataclass_from_dict(class_, dictionary: dict[str, str | float | int]) -> dict[str, str | float | int]:
    """Converts a dictionary to a dataclass instance.

    Args:
        class_: The dataclass type.
        dictionary: The dictionary to convert.

    Returns:
        Union[dict[str, Union[str, float, int]], dataclasses.dataclass]: The converted dataclass instance or the
            original dictionary.
    """
    try:
        field_types: dict = {f.name: f.type for f in dataclasses.fields(class_)}
        return class_(**{f: dataclass_from_dict(field_types.get(f), dictionary.get(f)) for f in dictionary})
    except Exception:
        return dictionary  # The object is not a dataclass field
