from shallowwater import __version__, backend_info


def test_version_and_backend_info_are_public():
    info = backend_info()

    assert __version__
    assert info["shallowwater"] == __version__
    assert info["backend"] in {"numpy", "numba"}
    assert isinstance(info["numba_available"], bool)
