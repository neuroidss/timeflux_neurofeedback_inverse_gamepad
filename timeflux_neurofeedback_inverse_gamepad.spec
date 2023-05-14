# -*- mode: python ; coding: utf-8 -*-

import sys
if sys.platform.startswith('win'):
#  site_packages='C:/Python39/Lib/site-packages'
  site_packages='c:/hostedtoolcache/windows/python/3.9.13/x64/lib/site-packages'
  binaries = [
#	(site_packages+"/_libsuinput.cpython-39-x86_64-linux-gnu.so", "."),
#    (site_packages+"/brainflow/lib/", "brainflow/lib"),
#    (site_packages+"/pylsl/lib/", "pylsl/lib"),
#    (site_packages+"/pyzmq.libs/", "pyzmq.libs"),
#    (site_packages+"/tables/", "tables"),
#    (site_packages+"/opencv_python.libs/", "opencv_python.libs"),
    ]
  datas=[
    (site_packages, "."),
    (site_packages+"/mne", "mne"),
    (site_packages+"/timeflux", "timeflux"),
    (site_packages+"/timeflux_neurofeedback_inverse_gamepad", "timeflux_neurofeedback_inverse_gamepad"),
    (site_packages+"/zmq", "zmq"),
    (site_packages+"/pythonosc", "pythonosc"),
    (site_packages+"/pyedflib", "pyedflib"),
    (site_packages+"/pylsl", "pylsl"),
#    (site_packages+"/evdev", "evdev"),
    (site_packages+"/xarray", "xarray"),
    (site_packages+"/timeflux_ui", "timeflux_ui"),
    (site_packages+"/aiohttp", "aiohttp"),
    (site_packages+"/multidict", "multidict"),
    (site_packages+"/timeflux_dsp", "timeflux_dsp"),
    (site_packages+"/scipy", "scipy"),
    (site_packages+"/gephistreamer", "gephistreamer"),
    (site_packages+"/pyvjoy", "pyvjoy"),
#    (site_packages+"/uinput", "uinput"),
    (site_packages+"/yarl", "yarl"),
    (site_packages+"/ws4py", "ws4py"),
    (site_packages+"/async_timeout", "async_timeout"),
    (site_packages+"/aiosignal", "aiosignal"),
    (site_packages+"/frozenlist", "frozenlist"),
    (site_packages+"/mne_connectivity", "mne_connectivity"),
#    ("/usr/lib/python3.9/timeit.py", "timeit"),
#    ("C:/Python39/Lib/timeit.py", "timeit"),
    ("c:/hostedtoolcache/windows/python/3.9.13/x64/lib/timeit.py", "timeit"),
    (site_packages+"/tqdm", "tqdm"),
    (site_packages+"/pyformulas", "pyformulas"),
    (site_packages+"/cv2", "cv2"),
    (site_packages+"/PIL", "PIL"),
    (site_packages+"/timeflux_brainflow", "timeflux_brainflow"),
    (site_packages+"/brainflow", "brainflow"),
    (site_packages+"/nptyping", "nptyping"),
    (site_packages+"/typish", "typish"),
    (site_packages+"/pooch", "pooch"),
    ]
else:
#  site_packages='./env_timeflux_coherence_circle/lib/python3.9/site-packages'
  site_packages='/opt/hostedtoolcache/Python/3.9.13/x64/lib/python3.9/site-packages'
#  site_packages='/root/.pyenv/versions/3.9.13/lib/python3.9/site-packages'
  
  binaries = [
    (site_packages+"/_libsuinput.cpython-39-x86_64-linux-gnu.so", "."),
    (site_packages+"/brainflow/lib/", "brainflow/lib"),
#    (site_packages+"/pylsl/lib/", "pylsl/lib"),
    ("/usr/lib/liblsl.so", "pylsl/lib"),
    (site_packages+"/pyzmq.libs/", "pyzmq.libs"),
    (site_packages+"/tables/", "tables"),
    (site_packages+"/opencv_python.libs/", "opencv_python.libs"),
    ]
  datas=[
    (site_packages, "."),
    (site_packages+"/mne", "mne"),
    (site_packages+"/timeflux", "timeflux"),
    (site_packages+"/timeflux_neurofeedback_inverse_gamepad", "timeflux_neurofeedback_inverse_gamepad"),
    (site_packages+"/zmq", "zmq"),
    (site_packages+"/pythonosc", "pythonosc"),
    (site_packages+"/pyedflib", "pyedflib"),
    (site_packages+"/pylsl", "pylsl"),
    (site_packages+"/evdev", "evdev"),
    (site_packages+"/xarray", "xarray"),
    (site_packages+"/timeflux_ui", "timeflux_ui"),
    (site_packages+"/aiohttp", "aiohttp"),
    (site_packages+"/multidict", "multidict"),
    (site_packages+"/timeflux_dsp", "timeflux_dsp"),
    (site_packages+"/scipy", "scipy"),
    (site_packages+"/gephistreamer", "gephistreamer"),
#    (site_packages+"/pyvjoy", "pyvjoy"),
    (site_packages+"/uinput", "uinput"),
    (site_packages+"/yarl", "yarl"),
    (site_packages+"/ws4py", "ws4py"),
    (site_packages+"/async_timeout", "async_timeout"),
    (site_packages+"/aiosignal", "aiosignal"),
    (site_packages+"/frozenlist", "frozenlist"),
    (site_packages+"/mne_connectivity", "mne_connectivity"),
#    ("/usr/lib/python3.9/timeit.py", "timeit"),
    ("/opt/hostedtoolcache/Python/3.9.15/x64/lib/python3.9/timeit.py", "timeit"),
#    ("/root/.pyenv/versions/3.9.15/lib/python3.9/timeit.py", "timeit"),
    (site_packages+"/tqdm", "tqdm"),
    (site_packages+"/pyformulas", "pyformulas"),
    (site_packages+"/cv2", "cv2"),
    (site_packages+"/PIL", "PIL"),
    (site_packages+"/timeflux_brainflow", "timeflux_brainflow"),
    (site_packages+"/brainflow", "brainflow"),
    (site_packages+"/nptyping", "nptyping"),
    (site_packages+"/typish", "typish"),
    (site_packages+"/pooch", "pooch"),
    ]

block_cipher = None


a = Analysis(
    ['timeflux_neurofeedback_inverse_gamepad.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='timeflux_neurofeedback_inverse_gamepad',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='timeflux_neurofeedback_inverse_gamepad',
)
