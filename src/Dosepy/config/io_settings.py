# This module is used to read and write the settings.toml file.

#TODO: Change app_controller to use the settings module to get the ROI size
#TODO: Create a GUI to change the settings

import tomlkit
import pathlib

from pydantic import BaseModel, Field, ValidationError


class Settings(BaseModel):
    """ Class to store the settings from the settings.toml file. An instance class is created using the load_settings function"""
    
    roi_size_h: float = Field(gt=0)  # Horizontal size of the ROI in mm, must be greater than 0
    roi_size_v: float = Field(gt=0)  # Vertical size of the ROI in mm, must be greater than 0

    def get_calib_roi_size(self) -> tuple:
        return (self.roi_size_h, self.roi_size_v)


def load_settings() -> Settings:
    path = pathlib.Path(__file__).parent.parent / "config" / "settings.toml"
    if not path.exists():
        # Create the file if it does not exist
        path.touch()
        # Create the default settings
        create_default_settings(path)

    with open(path) as fp:
        raw_settings = tomlkit.load(fp)

    # Handle exceptions if the settings toml file is not well formatted
    try:
        settings = Settings(**raw_settings["user"])
        return settings
    
    except ValidationError as e:
        # If the settings file is not well formatted, create a new one
        create_default_settings(path)
        print(e)
        print("The settings file was not well formatted. A new one was created with default values.")
        return load_settings()


def create_default_settings(path) -> None:
    # Used by load_settings if the settings.toml file does not exist or is not well formatted
    settings = tomlkit.document()
    settings.add(tomlkit.comment("This file holds the configuration for the application"))
    settings.add(tomlkit.nl())

    user = tomlkit.table()
    user.add(tomlkit.comment("ROI size in mm"))
    user.add("roi_size_h", 8.0)  # Default value: 8 millimeters
    user.add("roi_size_v", 8.0)
    
    settings.add("user", user)

    with open(path, mode = "wt", encoding="utf-8") as fp:
        tomlkit.dump(settings, fp)
