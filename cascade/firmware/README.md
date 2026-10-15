# Cascade firmware (AM2732R R5F of the AWR2243-2X-CAS-EVM)

`am273x_cascade_param.xer5f` is TI's `ti/utils/test/cascade/am273x` test from the
**mmWave MCU+ SDK 04.04.00.01**, changed so that every radar parameter comes from the
`gUserCfg` block (`src/cascade_user_cfg.h`) which DARTscope writes over JTAG before the
firmware runs. A change of settings therefore needs no rebuild; rebuild only for changes
beyond them.

What differs from the SDK test (`src/`):

| | SDK test | here |
|---|---|---|
| profile | hardcoded in `Mmwave_populateDefaultProfileCfg` (77 GHz, 11 MHz/µs, 128 samples @ 20 Msps) | from `gUserCfg` |
| chirps | `TEST_NUM_CHIRPS` identical chirps, TX mask 7 | 1-12 chirps per loop, a TX mask per chirp and device |
| frame | `TEST_NUM_LOOPS`, `TEST_NUM_FRAMES` | loops, period and frames from `gUserCfg`; the radar stops itself |
| CSI ping/pong buffers | `TEST_NUM_ADC_SAMPLES` | sized for 512 samples; the LVDS session for the actual count |
| status | test log only | `gUserCfg.state/errStage/errCode/framesDone` |

The HealthLabVSB repository's `common.c` changed `PROFILE_0_*` macros that frame mode never
reads; it ran the SDK profile above. Its 128 chirp configurations per device also overflow
the mmWave library's 8 KB chirp heap (`MMWAVE_HEAP_MEM_SIZE`), so the radar was never
configured with them.

## Rebuilding

The SDK's installers are Windows/Linux x86 programs. On a Mac they run in a Lima VM
(Ubuntu arm64, x86 binaries under qemu-user; Rosetta fails there with this kernel):

1. `brew install lima`, a VM with `mounts: [{location: <this folder>, writable: true}]`
2. in it: `dpkg --add-architecture i386 amd64`, `apt install qemu-user qemu-user-binfmt make libc6:i386 libc6:amd64 libstdc++6:amd64 libtinfo6:amd64` (and a link `libtinfo.so.5 -> libtinfo.so.6`)
3. `qemu-i386 -L / ./mmwave_mcuplus_sdk_04_04_00_01-Linux-x86-Install.bin --mode unattended --prefix ~/ti/mmwave_mcuplus_sdk_04_04_00_01`, then the bundled `mcu_plus_sdk_am273x_08_05_00_24-linux-x64-installer.run` and `sysconfig-1.14.0_2667-setup.run` (prefix `.../sysconfig_1.14.0`), and `ti_cgt_armllvm_2.1.2.LTS_linux-x64_installer.bin` into the same folder
4. `scripts/unix/setenv.sh` with `CCS_INSTALL_PATH` and `R5F_CLANG_INSTALL_PATH` pointing there

`build_firmware.sh` then copies `src/` over the SDK test and runs `make test` in the VM.
On Windows, the SDK's `setenv.bat` + `make` in `ti/utils/test/cascade` do the same.
