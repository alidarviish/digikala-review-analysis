import pytest

from digikala_reviews.resources import available_memory_gb, peak_memory_gb, peak_memory_gb_if_known


def test_available_memory_read_from_meminfo(tmp_path):
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal:  5000000 kB\nMemFree:  1000000 kB\nMemAvailable:  4194304 kB\n")
    assert available_memory_gb(meminfo) == pytest.approx(4.0)


def test_missing_line_is_an_error(tmp_path):
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal:  5000000 kB\n")
    with pytest.raises(RuntimeError):
        available_memory_gb(meminfo)


def test_peak_memory_is_positive():
    assert peak_memory_gb() > 0


def test_peak_memory_read_from_process_status(tmp_path):
    status = tmp_path / "status"
    status.write_text("Name:  python\nVmPeak:  9000000 kB\nVmHWM:  2097152 kB\nVmRSS:  1000 kB\n")
    assert peak_memory_gb(status) == pytest.approx(2.0)


def test_unknown_peak_memory_is_none(tmp_path):
    status = tmp_path / "status"
    status.write_text("Name:  python\nVmRSS:  1000 kB\n")  # no VmHWM, as on molab
    assert peak_memory_gb_if_known(status) is None
    status.write_text("VmHWM:  2097152 kB\n")
    assert peak_memory_gb_if_known(status) == 2.0
