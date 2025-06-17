import pytest
from Dosepy import bed
from pathlib import Path


# Test the bed module

# Test: TypeError if path_to_file is not str or Path
def test_load_type_error():
    with pytest.raises(TypeError):
        bed.load(123)

# Test: FileNotFoundError if file does not exist
def test_load_file_not_found():
    with pytest.raises(FileNotFoundError):
        bed.load("/non/existent/file.dcm")

# Test: ValueError if file is not a valid DICOM (missing DICM magic)
#def test_load_invalid_dicom():

