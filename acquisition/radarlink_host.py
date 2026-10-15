"""
acquisition/radarlink_host.py

Runs under a 32-bit Python (Radar/.py32) and holds TI's RadarLinkDLL.dll - the library
mmWave Studio drives the radar with - so the 64-bit application can call it without
mmWave Studio running. TI ships the library only as a 32-bit DLL, in every mmWave
Studio version, and a 64-bit process cannot load it.

Protocol, one JSON object per line:
    stdin   {"id": 7, "fn": "RadarLinkImpl_SOPControl", "args": [["u32", 1], ["i32", 2]]}
    stdout  {"id": 7, "ret": 0}                       or {"id": 7, "error": "..."}
    stdout  {"event": {"dev": 1, "msg": 0, "sbid": 20480, "payload": "0a0b..."}}
Argument types: u8 u16 u32 i8 i16 i32 char, "buf" (hex string, passed as a pointer to
a copy - a struct), "callback" (the async-event handler, only for RadarLinkImpl_PowerOn).
Every function of the DLL is cdecl (AR1xController's DllImport attributes).

Run: .py32\\python.exe -u radarlink_host.py <folder of RadarLinkDLL.dll>

Authors: Daniel Barvik, Dan Hruby, and AI
"""
import ctypes
import json
import os
import sys
import threading

TYPES = {"u8": ctypes.c_uint8, "u16": ctypes.c_uint16, "u32": ctypes.c_uint32,
         "i8": ctypes.c_int8, "i16": ctypes.c_int16, "i32": ctypes.c_int32,
         "char": ctypes.c_char}

# void __stdcall (uint32 DeviceMap, uint16 msgId, uint16 sbId, void *data) - the handler
# RadarLinkImpl_PowerOn registers (AR1xController's ScriptOps+SPIAsyncEventCallFunc).
# sbId is mmWaveLink's unique sub-block id, e.g. 0x5000 MSS power-up done, 0x5001 BSS
# (RF) power-up done, 0x1004 RF init calibration status.
EVENT_CB = ctypes.WINFUNCTYPE(None, ctypes.c_uint32, ctypes.c_uint16, ctypes.c_uint16,
                              ctypes.c_void_p)
# the event structs are at most a few dozen bytes; the length is not passed, so a fixed
# window of the DLL's receive buffer is forwarded
PAYLOAD_BYTES = 64

_out_lock = threading.Lock()


def emit(obj):
    with _out_lock:
        sys.stdout.write(json.dumps(obj) + "\n")
        sys.stdout.flush()


def _on_event(dev, msg, sbid, data):
    try:
        payload = ctypes.string_at(data, PAYLOAD_BYTES) if data else b""
    except Exception:
        payload = b""
    emit({"event": {"dev": int(dev), "msg": int(msg), "sbid": int(sbid), "payload": payload.hex()}})


_callback = EVENT_CB(_on_event)      # kept alive for the life of the process


def main():
    folder = sys.argv[1]
    os.add_dll_directory(folder)
    dll = ctypes.CDLL(os.path.join(folder, "RadarLinkDLL.dll"))
    emit({"ready": True, "dll": os.path.join(folder, "RadarLinkDLL.dll"),
          "bits": ctypes.sizeof(ctypes.c_void_p) * 8})
    keep = []
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req = json.loads(line)
        rid = req.get("id")
        if req.get("fn") == "__exit__":
            emit({"id": rid, "ret": 0})
            break
        try:
            fn = getattr(dll, req["fn"])
            fn.restype = ctypes.c_int32
            args, argtypes = [], []
            for kind, value in req.get("args", []):
                if kind == "buf":
                    b = ctypes.create_string_buffer(bytes.fromhex(value), len(value) // 2)
                    keep.append(b)
                    args.append(ctypes.cast(b, ctypes.c_void_p))
                    argtypes.append(ctypes.c_void_p)
                elif kind == "callback":
                    args.append(_callback)
                    argtypes.append(EVENT_CB)
                elif kind == "char":
                    args.append(ctypes.c_char(bytes([int(value) & 0xFF])))
                    argtypes.append(ctypes.c_char)
                else:
                    args.append(TYPES[kind](int(value)))
                    argtypes.append(TYPES[kind])
            fn.argtypes = argtypes
            ret = fn(*args)
            emit({"id": rid, "ret": int(ret)})
        except Exception as e:
            emit({"id": rid, "error": f"{type(e).__name__}: {e}"})


if __name__ == "__main__":
    main()
