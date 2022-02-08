from ml_downscaling_emulator.data.moose import moose_path

def test_moose_path():
    assert moose_path(variable="pr", year=1981) == "moose:crum/mi-bb171/apb.pp"
