from ml_downscaling_emulator.data.moose import moose_path, select_query

def test_moose_path():
    assert moose_path(variable="pr", year=1981) == "moose:crum/mi-bb171/apb.pp"

def test_select_query():
    year = 1981
    variable = "1.5mtmean"

    expected = """
begin
    yr=1980
    mon=12
    stash=3236
    lbproc=128
end

begin
    yr=1981
    mon=[1..11]
    stash=3236
    lbproc=128
end
""".lstrip()

    assert select_query(year, variable) == expected
