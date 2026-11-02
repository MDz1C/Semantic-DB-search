# PyInstaller spec: one-folder, windowed Windows build of Fundscope.
#
#   .venv\Scripts\python.exe -m PyInstaller --noconfirm --clean packaging\semdb.spec
#       --distpath dist --workpath build\pyinstaller
#
# packaging\build.ps1 runs this, then the frozen --selftest and Inno Setup.  What is bundled
# or excluded, and why, is described in docs\ARCHITECTURE.md sections 13 and 14.

import re
import sys
from importlib.metadata import distribution
from pathlib import Path

import pefile
from PyInstaller.utils.hooks import collect_submodules
from PyInstaller.utils.win32 import versioninfo as vi

ROOT = Path(SPECPATH).parent
SRC = ROOT / "src"
MODEL = ROOT / "models" / "paraphrase-multilingual-MiniLM-L12-v2"
ASSETS = ROOT / "assets"
LICENSES = Path(SPECPATH) / "licenses"
NAME = "Fundscope"
PUBLISHER = "Manuel Dzimah"

_package = (SRC / "semdb" / "__init__.py").read_text(encoding="utf-8")
VERSION = re.search(r'__version__ = "([^"]+)"', _package).group(1)
APP_NAME = re.search(r'APP_NAME = "([^"]+)"', _package).group(1)

_inputs = {
    MODEL / "model.onnx": "tools/fetch_model.py",
    MODEL / "tokenizer.json": "tools/fetch_model.py",
    ASSETS / "icon.ico": "tools/make_icon.py",
    ASSETS / "sample" / "sample_funds.xlsx": "tools/make_sample_data.py",
    ROOT / "THIRD_PARTY_NOTICES.md": "the repository",
}
_missing = [f"{path.relative_to(ROOT)} (from {origin})" for path, origin in _inputs.items() if not path.is_file()]
if _missing:
    raise SystemExit("Missing build inputs:\n  " + "\n  ".join(_missing))

# Nothing here is used at runtime.  Leaving them out keeps every HTTP client, the TLS stack, the
# standard library's servers and Qt's network stack out of the bundle (R1: no network access, no
# server), and keeps the download small.  The standard library imports ssl only optionally.
EXCLUDES = [
    "tkinter", "_tkinter", "unittest", "pydoc", "pydoc_data",
    "huggingface_hub", "hf_xet", "httpx", "httpx2", "httpcore", "httpcore2",
    "requests", "urllib3", "fsspec", "yaml", "onnx",
    "ssl", "_ssl", "http.server", "socketserver", "xmlrpc.server", "wsgiref",
    *(f"PySide6.{module}" for module in (
        "QtNetwork", "QtQml", "QtQmlFeatures", "QtQuick", "QtQuickControls2", "QtQuickTest",
        "QtQuickWidgets", "QtQuick3D", "QtWebEngineCore", "QtWebEngineQuick", "QtWebEngineWidgets",
        "QtWebChannel", "QtWebSockets", "QtPdf", "QtPdfWidgets", "QtOpenGL", "QtOpenGLWidgets",
        "QtSql", "QtTest", "QtDesigner", "QtHelp", "QtUiTools", "QtDBus", "QtConcurrent", "QtXml",
        "QtLabsStyleKit", "QtPrintSupport",
    )),
]

# Libraries that may still be pulled in as dependencies but must not ship.  onnxruntime.dll is
# the C API build; the Python binding (onnxruntime_pybind11_state.pyd) contains the whole runtime
# and never loads it (no import entry, no LoadLibrary name), so it would be 19 MB of dead weight.
_DROPPED_LIBRARY = re.compile(r"(qt6(network|qml\w*|quick\w*|pdf\w*|opengl\w*)|opengl32sw|onnxruntime)\.dll")
# Qt plugins kept, per plugin folder; every other folder is dropped (tls, networkinformation,
# qmltooling, sqldrivers, generic = TUIO touch input over UDP, platforminputcontexts...).
_KEPT_PLUGINS = {
    "platforms": {"qwindows.dll"},
    "styles": None,  # all of them
    "imageformats": {"qsvg.dll", "qico.dll"},  # PNG support is built into QtGui
    "iconengines": {"qsvgicon.dll"},
}


def _keep(dest: str) -> bool:
    path = dest.replace("\\", "/").lower()
    name = path.rsplit("/", 1)[-1]
    if _DROPPED_LIBRARY.fullmatch(name):
        return False
    if "/plugins/" in path:
        folder = path.split("/plugins/", 1)[1].split("/", 1)[0]
        if folder not in _KEPT_PLUGINS:
            return False
        kept = _KEPT_PLUGINS[folder]
        return kept is None or name in kept
    if "/translations/" in path:
        return name.endswith("_en.qm")
    return True


def _imports(binary: str) -> list[str]:
    pe = pefile.PE(binary, fast_load=True)
    pe.parse_data_directories(directories=[
        pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_IMPORT"],
        pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_DELAY_IMPORT"],
    ])
    entries = getattr(pe, "DIRECTORY_ENTRY_IMPORT", []) + getattr(pe, "DIRECTORY_ENTRY_DELAY_IMPORT", [])
    pe.close()
    return [entry.dll.decode("ascii", "replace").lower() for entry in entries]


def _metadata_licenses(dist_name: str) -> list[tuple[str, str]]:
    """Licence files and SBOMs a wheel ships in its .dist-info folder -> licenses/<name>/."""
    dist = distribution(dist_name)
    entries = []
    for file in dist.files or []:
        parts = file.parts
        if len(parts) < 2 or not parts[0].endswith(".dist-info"):
            continue
        nested = parts[1] in ("licenses", "sboms")
        if nested or parts[-1].upper().startswith(("LICENSE", "LICENCE", "COPYING", "NOTICE")):
            folder = Path("licenses", dist.metadata["Name"], *(parts[2:-1] if nested else ()))
            entries.append((str(file.locate()), str(folder)))
    return entries


_onnxruntime = Path(distribution("onnxruntime").locate_file("onnxruntime"))
LICENSE_DATAS = [
    (str(ROOT / "LICENSE"), "."),
    (str(ROOT / "THIRD_PARTY_NOTICES.md"), "."),
    (str(Path(sys.base_prefix) / "LICENSE.txt"), "licenses/Python"),
    (str(LICENSES / "LGPL-3.0.txt"), "licenses/Qt"),
    (str(LICENSES / "GPL-3.0.txt"), "licenses/Qt"),
    (str(LICENSES / "Apache-2.0.txt"), "licenses/tokenizers"),
    (str(LICENSES / "Apache-2.0.txt"), f"licenses/{MODEL.name}"),
    (str(LICENSES / "Lucide-ISC.txt"), "licenses/Lucide"),
    (str(_onnxruntime / "LICENSE"), "licenses/onnxruntime"),
    (str(_onnxruntime / "ThirdPartyNotices.txt"), "licenses/onnxruntime"),
    *(entry for dist in ("numpy", "RapidFuzz", "python-calamine", "XlsxWriter", "tokenizers", "pyinstaller")
      for entry in _metadata_licenses(dist)),
]

sys.path.insert(0, str(SRC))  # so collect_submodules can see the package before Analysis runs

a = Analysis(
    [str(Path(SPECPATH) / "launcher.py")],
    pathex=[str(SRC)],
    datas=[(str(MODEL), f"models/{MODEL.name}"), (str(ASSETS), "assets"), *LICENSE_DATAS],
    hiddenimports=collect_submodules("semdb"),
    excludes=EXCLUDES,
    noarchive=False,
)

_dropped = {Path(dest).name.lower() for dest, _, _ in a.binaries if not _keep(dest)}
a.binaries = [entry for entry in a.binaries if _keep(entry[0])]
a.datas = [entry for entry in a.datas if _keep(entry[0])]

# A kept library that links against a dropped one would only fail on the user's machine.
_broken = [
    f"{dest} -> {dll}"
    for dest, src, _ in a.binaries
    if dest.lower().endswith((".dll", ".pyd"))
    for dll in _imports(src)
    if dll in _dropped
]
if _broken:
    raise SystemExit("Bundled libraries depend on dropped ones:\n  " + "\n  ".join(_broken))

_numbers = tuple(int(part) for part in VERSION.split(".")) + (0,) * (4 - len(VERSION.split(".")))
VERSION_INFO = vi.VSVersionInfo(
    ffi=vi.FixedFileInfo(filevers=_numbers, prodvers=_numbers, mask=0x3F, flags=0x0, OS=0x40004,
                         fileType=0x1, subtype=0x0, date=(0, 0)),
    kids=[
        vi.StringFileInfo([vi.StringTable("040904B0", [
            vi.StringStruct("CompanyName", PUBLISHER),
            vi.StringStruct("FileDescription", APP_NAME),
            vi.StringStruct("FileVersion", VERSION),
            vi.StringStruct("InternalName", NAME),
            vi.StringStruct("LegalCopyright", f"Copyright (c) 2026 {PUBLISHER}. MIT License."),
            vi.StringStruct("OriginalFilename", f"{NAME}.exe"),
            vi.StringStruct("ProductName", APP_NAME),
            vi.StringStruct("ProductVersion", VERSION),
        ])]),
        vi.VarFileInfo([vi.VarStruct("Translation", [0x0409, 1200])]),
    ],
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=NAME,
    console=False,
    upx=False,
    strip=False,
    icon=str(ASSETS / "icon.ico"),
    version=VERSION_INFO,
)

coll = COLLECT(exe, a.binaries, a.datas, name=NAME, upx=False, strip=False)
