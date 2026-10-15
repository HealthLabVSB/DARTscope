# -*- mode: python ; coding: utf-8 -*-
#
# DARTscope.exe, from the Radar folder, in the clean build environment of
# requirements-build.txt (not Anaconda - its MKL-based numpy makes a 300 MB exe):
#   .buildenv\Scripts\python -m PyInstaller DARTscope.spec --noconfirm --clean
#
# Besides the program it carries what recording without mmWave Studio needs: the helper
# script that loads TI's 32-bit RadarLinkDLL and the 32-bit Python it runs under (.py32/,
# python.org's 3.12 embeddable zip). mmWave Studio itself only has to be installed - the
# radar's library and firmware come from C:\ti\mmwave_studio_02_01_01_00.
# Check a build: set DARTSCOPE_SELFTEST=report.json and start the exe (acquisition/diagnostics.py).
import re

from PyInstaller.utils.win32.versioninfo import (FixedFileInfo, StringFileInfo, StringStruct,
                                                 StringTable, VarFileInfo, VarStruct,
                                                 VSVersionInfo)

with open('version.py', encoding='utf-8') as f:
    VERSION = re.search(r'__version__\s*=\s*"([^"]+)"', f.read()).group(1)
_nums = tuple((list(map(int, VERSION.split('.'))) + [0, 0, 0, 0])[:4])
version_info = VSVersionInfo(
    ffi=FixedFileInfo(filevers=_nums, prodvers=_nums, mask=0x3F, flags=0x0, OS=0x40004,
                      fileType=0x1, subtype=0x0, date=(0, 0)),
    kids=[StringFileInfo([StringTable('040904B0', [
              StringStruct('ProductName', 'DARTscope'),
              StringStruct('FileDescription', 'DARTscope - mmWave radar recording and analysis'),
              StringStruct('FileVersion', VERSION),
              StringStruct('ProductVersion', VERSION),
              StringStruct('OriginalFilename', 'DARTscope.exe'),
              StringStruct('InternalName', 'DARTscope')])]),
          VarFileInfo([VarStruct('Translation', [0x0409, 1200])])])

a = Analysis(
    ['DCA_processing.py'],
    pathex=[],
    binaries=[],
    datas=[('docs/MANUAL.md', 'docs'), ('icon.png', '.'), ('icon.svg', '.'),
           ('acquisition/radarlink_host.py', 'acquisition'),   # run by the 32-bit Python
           ('.py32', 'py32'),                                  # the 32-bit Python itself
           ('cascade/jtag_cfg.js', 'cascade'),                 # cascade radar: JTAG script,
           ('cascade/AM273x_XDS110.ccxml', 'cascade'),         # its target configuration
           ('cascade/firmware/am273x_cascade_param.xer5f', 'cascade/firmware')],  # and firmware
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # Built from an Anaconda base, the analysis follows other libraries' optional imports
    # into packages DARTscope never uses (torch alone is gigabytes). The program needs
    # numpy, scipy, matplotlib (TkAgg), Pillow and scikit-learn (AI tab); none of these.
    excludes=['torch', 'torchvision', 'torchaudio', 'tensorflow', 'keras',
              'PyQt5', 'PyQt6', 'PySide2', 'PySide6', 'qtpy', 'sip',
              'IPython', 'ipykernel', 'ipywidgets', 'jupyter', 'jupyter_client', 'jupyter_core',
              'jupyterlab', 'notebook', 'nbformat', 'nbconvert', 'jedi', 'parso',
              'pytest', '_pytest', 'black', 'sphinx', 'docutils', 'numba', 'llvmlite',
              'dask', 'distributed', 'bokeh', 'panel', 'holoviews', 'pandas', 'pyarrow',
              'sympy', 'tables', 'h5py', 'openpyxl', 'lxml', 'cryptography', 'bcrypt', 'nacl',
              'paramiko', 'sqlalchemy', 'botocore', 'boto3', 's3fs', 'fsspec', 'zmq', 'tornado',
              'astropy', 'statsmodels', 'seaborn', 'plotly', 'cv2', 'skimage', 'imageio',
              'networkx', 'xarray', 'clr', 'pythonnet', 'docx'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='DARTscope',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['icon.ico'],
    version=version_info,
)
