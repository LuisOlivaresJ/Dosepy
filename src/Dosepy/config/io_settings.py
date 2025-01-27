"""
This module is used to manage settings for the GUI application (Dosepy.app).
It uses an instance of the Settings class constructed with the load_settings() function.

A settings.toml file is used to store setting on disk. Existing "key-value" pairs can be
modified as needed. 

New key-value pairs must not be directly added to settings.toml file,
instead add them as attriutes of the class Settings and update the 
_create_default_settings() function.
"""

#TODO: Change app_controller to use the settings module to get the ROI size
#TODO: Create a GUI to change the settings

import tomlkit
import pathlib

from pydantic import BaseModel, Field, ValidationError

# Path to the settings.toml file
toml_file_path = pathlib.Path(__file__).parent.parent / "config" / "settings.toml"

class Settings(BaseModel):
    """
    Class used to manage settings.

    Notes
    -----
    * An instance of this class must be created using the load_settings() function (see below)
    * Every set_some_attribute() method must save to settings.toml file.
    """
    
    roi_automatic: str = "Disable"  # Used to create an automatic roi size.
    roi_size_h: float = Field(default = 8, gt=0)  # Horizontal size of the ROI in mm, must be greater than 0
    roi_size_v: float = Field(default = 8, gt=0)  # Vertical size of the ROI in mm, must be greater than 0

    channel: str = "red"  # Channel to use for the dose calculation
    fit_function: str = "Rational"  # Fit function to use for the dose calculation

    lateral_correction: bool = False  # If lateral correction is enabled


    def get_roi_automatic(self) -> str:
        return self.roi_automatic

    def get_calib_roi_size(self) -> tuple:
        return (self.roi_size_h, self.roi_size_v)
    
    def get_channel(self) -> str:
        return self.channel
    
    def get_fit_function(self) -> str:
        return self.fit_function
    
    def get_lateral_correction(self) -> bool:
        return self.lateral_correction
    

    def set_roi_automatic(self, roi_automatic: bool) -> None:
        """ 
        Set the ROI automatic in the settings.toml file 

        Parameters
        ----------
            roi_automatic : bool 
            If the ROI size is has to be calculated automatically
        """
        # Update the ROI automatic
        self.roi_automatic = roi_automatic
        seetings = _load_toml_file()
        seetings["user"]["roi_automatic"] = roi_automatic
        # Save to the settings.toml file
        with open(toml_file_path, mode="wt", encoding="utf-8") as fp:
            tomlkit.dump(seetings, fp)
    

    def set_calib_roi_size(self, roi_size: tuple) -> None:
        """ 
        Set the ROI size in the settings.toml file 
        Parameters
        ----------
            roi_size : tuple 
            Tuple with two floats representing the horizontal and vertical size of the ROI in mm
        """
        # Handle exceptions if the tuple is not well formatted
        if not isinstance(roi_size, tuple) or len(roi_size) != 2:
            raise ValueError("The ROI size must be a tuple with two elements")
        
        try:
            # Update the ROI size
            self.roi_size_h, self.roi_size_v = roi_size
            seetings = _load_toml_file()
            seetings["user"]["roi_size_h"] = roi_size[0]
            seetings["user"]["roi_size_v"] = roi_size[1]
            # Save to the settings.toml file
            with open(toml_file_path, mode="wt", encoding="utf-8") as fp:
                tomlkit.dump(seetings, fp)

        except ValidationError as e:
            print(e)
            raise ValueError("The ROI size must be a tuple with two floats")
        

    def set_channel(self, channel: str) -> None:
        """ 
        Set the channel in the settings.toml file 
        Parameters
        ----------
            channel : str 
            String with the channel to use for the dose calculation
        """
        # Update the channel
        self.channel = channel
        seetings = _load_toml_file()
        seetings["user"]["channel"] = channel
        # Save to the settings.toml file
        with open(toml_file_path, mode="wt", encoding="utf-8") as fp:
            tomlkit.dump(seetings, fp)


    def set_fit_function(self, fit_function: str) -> None:
        """ 
        Set the fit function in the settings.toml file 
        Parameters
        ----------
            fit_function : str 
            Fit function to use for the dose calculation
        """
        # Update the fit function
        self.fit_function = fit_function
        seetings = _load_toml_file()
        seetings["user"]["fit_function"] = fit_function
        # Save to the settings.toml file
        with open(toml_file_path, mode="wt", encoding="utf-8") as fp:
            tomlkit.dump(seetings, fp)

    def set_lateral_correction(self, lateral_correction: bool) -> None:
        """ 
        Set the lateral correction in the settings.toml file 
        Parameters
        ----------
            lateral_correction : bool 
            If lateral correction is enabled
        """
        # Update the lateral correction
        self.lateral_correction = lateral_correction
        seetings = _load_toml_file()
        seetings["user"]["lateral_correction"] = lateral_correction
        # Save to the settings.toml file
        with open(toml_file_path, mode="wt", encoding="utf-8") as fp:
            tomlkit.dump(seetings, fp)
            

def load_settings() -> Settings:
    """ Load the settings from the settings.toml file """
    _create_toml_file_if_not_exists()
    raw_settings = _load_toml_file()

    # Handle exceptions if the settings toml file is not well formatted
    try:
        settings = Settings(**raw_settings["user"])
        return settings
    
    except ValidationError as e:
        # If the settings file is not well formatted, create a new one
        _create_default_settings(toml_file_path)
        print(e)
        print("The settings file was not well formatted. A new one was created with default values.")
        return load_settings()


def _load_toml_file() -> tomlkit.TOMLDocument:
    _create_toml_file_if_not_exists()
    with open(toml_file_path, mode="rt", encoding="utf-8") as fp:
        return tomlkit.load(fp)


def _create_toml_file_if_not_exists() -> None:
    if not toml_file_path.exists():
        # Create the file if it does not exist
        toml_file_path.touch()
        # Create the default settings
        _create_default_settings(toml_file_path)


def _create_default_settings(path) -> None:
    # Used by load_settings if the settings.toml file does not exist or is not well formatted
    settings = tomlkit.document()
    settings.add(tomlkit.comment("This file holds the configuration for the application"))
    settings.add(tomlkit.nl())

    user = tomlkit.table()
    user.add(tomlkit.comment("Automatic roi size"))
    user.add("roi_automatic", "Enable")
    user.add(tomlkit.comment("ROI size in mm"))
    user.add("roi_size_h", 8.0)  # Default value: 8 millimeters
    user.add("roi_size_v", 8.0)
    user.add(tomlkit.nl())
    user.add(tomlkit.comment("Channel to use for the dose calculation"))
    user.add("channel", "Red")
    user.add(tomlkit.nl())
    user.add(tomlkit.comment("Fit function to use for the dose calculation"))
    user.add("fit_function", "Rational")
    user.add(tomlkit.nl())
    user.add(tomlkit.comment("If lateral correction is enabled"))
    user.add("lateral_correction", False)
    
    settings.add("user", user)

    with open(path, mode = "wt", encoding="utf-8") as fp:
        tomlkit.dump(settings, fp)
