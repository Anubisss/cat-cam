import hashlib
import io
import subprocess
import urllib.error
from types import SimpleNamespace

import numpy as np
import pytest

from catcam import blackhole
from catcam.blackhole import (
    CHECKING,
    DOWNLOADING,
    FAILED,
    MISSING,
    OPENED,
    PAGE,
    READY,
    RESTART,
    SIGNER,
    WEBSITE,
    Failed,
    Status,
    VoiceSetup,
    fetch_installer,
    stage_for,
)

INSTALLER = b"not really an installer, but it stands in for one" * 1000

# What `pkgutil --check-signature` says about BlackHole2ch-0.7.1.pkg (trimmed).
PKGUTIL = f"""Package "BlackHole2ch-0.7.1.pkg":
   Status: signed by a developer certificate issued by Apple for distribution
   Notarization: trusted by the Apple notary service
   Certificate Chain:
    1. {SIGNER}
       Expires: 2027-02-01 22:12:15 +0000
       ------------------------------------------------------------------------
    2. Developer ID Certification Authority
       ------------------------------------------------------------------------
    3. Apple Root CA
"""


@pytest.fixture(autouse=True)
def _downloads_in_tmp(monkeypatch, tmp_path):
    """The button's download goes into the test's own folder, not the system's temp."""
    folder = tmp_path / "download"
    monkeypatch.setattr(blackhole.tempfile, "mkdtemp", lambda prefix: str(folder.mkdir() or folder))


class _Response(io.BytesIO):
    def __init__(self, data: bytes):
        super().__init__(data)
        self.headers = {"Content-Length": str(len(data))}


class _Mac:
    """Stands in for the network, pkgutil and `open`, and remembers what was asked."""

    def __init__(self, monkeypatch, data=INSTALLER, signer=SIGNER, online=True, pkgutil_ok=True):
        self.data, self.signer, self.online, self.pkgutil_ok = data, signer, online, pkgutil_ok
        self.opened, self.fetched = [], []
        monkeypatch.setattr(blackhole, "SHA256", hashlib.sha256(INSTALLER).hexdigest())
        monkeypatch.setattr(blackhole.urllib.request, "urlopen", self.urlopen)
        monkeypatch.setattr(blackhole.subprocess, "run", self.run)

    def urlopen(self, request, timeout):
        self.fetched.append(request.full_url)
        if not self.online:
            raise urllib.error.URLError("nodename nor servname provided, or not known")
        return _Response(self.data)

    def run(self, command, **kwargs):
        if command[0] == "pkgutil":
            output = PKGUTIL.replace(SIGNER, self.signer)
            return SimpleNamespace(returncode=0 if self.pkgutil_ok else 1, stdout=output, stderr="")
        if command[0] == "open":
            self.opened.append(command[1])
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        raise AssertionError(f"unexpected command {command}")


def test_stage_for_the_mac(tmp_path):
    devices = [{"name": "MacBook Pro Microphone", "max_input_channels": 1, "max_output_channels": 0},
               {"name": "MacBook Pro Speakers", "max_input_channels": 0, "max_output_channels": 2}]
    blackhole_2ch = {"name": "BlackHole 2ch", "max_input_channels": 2, "max_output_channels": 2}
    driver = tmp_path / "BlackHole2ch.driver"
    assert stage_for(devices, driver) == MISSING
    driver.mkdir()
    assert stage_for(devices, driver) == RESTART  # installed, not loaded yet
    assert stage_for(devices + [blackhole_2ch], driver) == READY


def test_fetch_installer_downloads_and_checks_it(monkeypatch, tmp_path):
    mac = _Mac(monkeypatch)
    progress, checked = [], []
    path = fetch_installer(tmp_path, progress.append, lambda: checked.append(True))
    assert mac.fetched == [blackhole.URL]
    assert path == tmp_path / "BlackHole2ch-0.7.1.pkg" and path.read_bytes() == INSTALLER
    assert progress[-1] == 1.0 and progress == sorted(progress) and checked == [True]
    assert mac.opened == []  # fetching never opens it


def test_a_wrong_checksum_is_deleted_not_opened(monkeypatch, tmp_path):
    mac = _Mac(monkeypatch, data=INSTALLER + b"tampered")
    with pytest.raises(Failed, match="isn't the BlackHole installer"):
        fetch_installer(tmp_path)
    assert list(tmp_path.iterdir()) == [] and mac.opened == []


@pytest.mark.parametrize("signer, ok", [("Developer ID Installer: Someone Else (ABCDE12345)", True),
                                        (SIGNER, False)])
def test_a_wrong_or_untrusted_signature_is_deleted_not_opened(monkeypatch, tmp_path, signer, ok):
    mac = _Mac(monkeypatch, signer=signer, pkgutil_ok=ok)
    with pytest.raises(Failed, match="isn't signed"):
        fetch_installer(tmp_path)
    assert list(tmp_path.iterdir()) == [] and mac.opened == []


def test_offline_says_so(monkeypatch, tmp_path):
    _Mac(monkeypatch, online=False)
    with pytest.raises(Failed, match="Are you online"):
        fetch_installer(tmp_path)


def test_signed_by_without_pkgutil(monkeypatch, tmp_path):
    def missing(*args, **kwargs):
        raise FileNotFoundError("pkgutil")

    monkeypatch.setattr(blackhole.subprocess, "run", missing)
    assert blackhole.signed_by(tmp_path / "x.pkg") is None


def _drawn(setup: VoiceSetup) -> np.ndarray:
    view = np.full((540, 960, 3), 120, np.uint8)
    setup.draw(view, 630, 70)
    return view


def _click(setup: VoiceSetup) -> bool:
    x, y, w, h = setup.button
    return setup.click(x + w // 2, y + h // 2)


def _finish(setup: VoiceSetup):
    setup._thread.join(timeout=5)
    assert not setup._thread.is_alive()


def test_the_button_installs_only_when_clicked(monkeypatch):
    mac = _Mac(monkeypatch)
    setup = VoiceSetup(MISSING)
    _drawn(setup)
    assert setup.button is not None
    assert mac.fetched == [] and setup._thread is None  # nothing happens on its own
    assert not setup.click(5, 5)  # a click elsewhere
    assert mac.fetched == []
    assert _click(setup)
    _finish(setup)
    assert setup.status == Status(OPENED)
    assert len(mac.opened) == 1 and mac.opened[0].endswith("/BlackHole2ch-0.7.1.pkg")
    _drawn(setup)
    assert setup.button is None  # no button once the installer is open


@pytest.mark.parametrize("trouble", [{"online": False}, {"data": b"tampered"},
                                     {"signer": "Developer ID Installer: Someone Else (ABCDE12345)"}])
def test_on_failure_the_button_offers_the_website(monkeypatch, trouble):
    mac = _Mac(monkeypatch, **trouble)
    setup = VoiceSetup(MISSING)
    _drawn(setup)
    _click(setup)
    _finish(setup)
    assert setup.status.stage == FAILED and setup.status.problem
    assert mac.opened == []  # nothing unverified is ever opened
    _drawn(setup)
    assert setup.button is not None
    _click(setup)
    assert mac.opened == [PAGE] and setup.status == Status(WEBSITE)


def test_a_failing_open_shows_as_a_failure(monkeypatch):
    mac = _Mac(monkeypatch)

    def run(command, **kwargs):
        if command[0] == "open":
            raise subprocess.CalledProcessError(1, command)
        return mac.run(command, **kwargs)

    monkeypatch.setattr(blackhole.subprocess, "run", run)
    setup = VoiceSetup(MISSING)
    _drawn(setup)
    _click(setup)
    _finish(setup)
    assert setup.status.stage == FAILED


def test_what_shows_at_each_stage():
    blank = np.full((540, 960, 3), 120, np.uint8)
    for stage in (MISSING, FAILED):
        setup = VoiceSetup(stage)
        assert np.any(_drawn(setup) != blank) and setup.button is not None
    for stage in (DOWNLOADING, CHECKING, OPENED, WEBSITE, RESTART):
        setup = VoiceSetup(stage)
        assert np.any(_drawn(setup) != blank) and setup.button is None, stage
    setup = VoiceSetup(READY)  # BlackHole is there: nothing at all
    assert np.array_equal(_drawn(setup), blank) and setup.button is None
    assert not setup.click(480, 270)


def test_the_download_bar_fills():
    setup = VoiceSetup(DOWNLOADING)
    setup.status = Status(DOWNLOADING, 0.1)
    some = _drawn(setup)
    setup.status = Status(DOWNLOADING, 0.9)
    more = _drawn(setup)
    assert np.any(some != more)


def test_the_preview_shows_the_button_under_the_voice_switch(monkeypatch):
    from catcam.app import Controls, Mouse, preview, toggle_rects
    from catcam.picker import Picker

    out = np.zeros((720, 1280, 3), np.uint8)
    cat = SimpleNamespace(name="Test")
    setup = VoiceSetup(MISSING)
    plain = preview(out, cat, False)
    with_button = preview(out, cat, False, voice_setup=setup)
    assert np.any(with_button != plain)
    x, y, w, h = toggle_rects(False)["voice"]
    bx, by, bw, bh = setup.button
    assert by > y + h and bx + bw <= x + w + 6  # below the Voice switch, not past its right end
    ready = VoiceSetup(READY)
    assert np.array_equal(preview(out, cat, False, voice_setup=ready), plain)

    # Clicking it in the window starts the download, and toggles no switch.
    clicked = []
    monkeypatch.setattr(setup, "click", lambda x, y: clicked.append((x, y)) or True)
    sprite = np.full((20, 20, 4), 255, np.uint8)
    picker = Picker([("A", sprite, np.zeros(3, np.uint8))])
    controls = Controls(picker, Mouse(picker), setup)
    controls.handle(ord("1"), None)  # leave the picker
    switches = (controls.tail, controls.head3d, controls.background, controls.voice)
    controls.mouse(4, bx + bw // 2, by + bh // 2, 0, None)
    controls.handle(-1, None)
    assert clicked == [(bx + bw // 2, by + bh // 2)]
    assert (controls.tail, controls.head3d, controls.background, controls.voice) == switches
