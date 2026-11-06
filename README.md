<!--
# SPDX-FileCopyrightText: Copyright 2025 Eden Emulator Project
# SPDX-License-Identifier: GPL-3.0-or-later

# SPDX-FileCopyrightText: 2018 yuzu Emulator Project
# SPDX-License-Identifier: GPL-2.0-or-later
-->
<!-- lang: en-GB -->

<h1 align="center">
  <br>
  <img src="dist/Orchid.png" alt="Orchid" width="200">
  <br>
  <b>Orchid</b>
  <br>
</h1>

<h4 align="center">A very experimental Switch 2 emulator. Orchid started from
<a href="https://github.com/zortos293/Eglantine">Eglantine</a> and has grown into its own project, with fixes aimed at
Mario Tennis Fever first, plus the shader, kernel and graphics work other games need.</h4>

> [!NOTE]
> Orchid is an independent project. Problems with these builds are not caused by the Eglantine
> or Eden developers, so please do not report them there.

<p align="center">
  <a href="#project-scope">Project scope</a> |
  <a href="#branches-and-builds">Branches and builds</a> |
  <a href="#what-orchid-adds">What Orchid adds</a> |
  <a href="#performance-and-compatibility">Performance</a> |
  <a href="#download">Download</a> |
  <a href="#build-the-newest-commit-yourself">Build it yourself</a> |
  <a href="#documentation">Documentation</a> |
  <a href="#license">License</a>
</p>

## Project scope

Orchid's support and development focus is on demonstrably improving these games:

- Mario Tennis Fever
- Both Hyrule Warriors games
- Pokémon Pokopia
- Star Fox
- Super Mario Bros. Wonder
- Yoshi and the Mysterious Book
- Donkey Kong Bananza
- Kirby Air Riders
- Kirby and the Forgotten Land

These are the project's target games; inclusion here does not mean a game is already fully
playable or free of issues. See [Performance and compatibility](#performance-and-compatibility)
for the currently reported results and their test conditions.

## Branches and builds

| Branch | Channel | Release name |
|---|---|---|
| `main` | stable | `v0.<minor>.<patch>`, regular release |

`.github/version` sets the minor version; CI counts the patch up. In 0.5 all branches were merged
into `main` (formerly `mario-tennis-fever`), including the experimental `mario-tennis-fever-perf`.

Commits land often, but they do not start a build. Release builds are started by hand
(Actions > release-binaries > Run workflow), so the newest release can be several commits behind
its branch. To run the newest commit, [build it yourself](#build-the-newest-commit-yourself).

## What Orchid adds

Compared with the Eglantine it started from:

- **Mario Tennis Fever:** startup crash fixed, the NVIDIA crash at match start fixed, and per-game
  defaults (low GPU accuracy, balanced fences).
- **Shaders:** many more Ampere instructions lowered (IMAD.WIDE, IADD3 carries, LEA.HI, F2I.U64,
  I2F from 64-bit pairs, bindless TEX modes and others), mainly from the Fire Emblem: Fortune's
  Weave bring-up.
- **System:** Switch 2 built-in fonts served from their own shared memory; kernel thread start and
  finalize races fixed.
- **Graphics:** query reports flushed before MME DMA reads, layered cross-format texture copies,
  separate surface and texture descriptor buffers, geometry shaders and framebuffers kept within
  device limits.
- **Experimental branch only:** a performance pass (buffer cache, puller, memory trap and kernel
  locking changes). Faster on some systems, less tested.
- **Updates:** release builds check this repository for updates and only offer builds of their
  own channel.

## Performance and compatibility

> [!NOTE]
> Measured by hand in November 2026 on one PC: AMD Ryzen 7 5800X3D, AMD Radeon RX 9070 XT,
> 32 GB RAM, CachyOS (Linux 7.x, Mesa RADV from git), 1x resolution scaling. Other hardware
> will differ.

| Game | FPS | Notes |
|---|---|---|
| Hyrule Warriors: Age of Imprisonment | 60 | after shader compilation |
| Donkey Kong Bananza | 30, 15-25 in heavy areas | 30 in lighter areas |
| Mario Kart World | 30-40 | lots of off-track asset glitches |
| Mario Tennis Fever | 60 | runs flawlessly |
| Yoshi and the Mysterious Book | | gameplay renders a black screen |
| Super Mario Bros. Wonder | 60 | runs flawlessly |
| Star Fox | 30 | crashes very quickly |
| Splatoon Raiders | 30 | in gameplay |
| Pokémon Pokopia | 60 | in gameplay |
| Pokémon Legends: Z-A | 30 | in gameplay, the Switch 1 frame cap |
| Orbitals | | does not launch |
| Nintendo Switch Sports Resort | 60 | broken lighting, broken controller selection |
| Nintendo Switch 2 Welcome Tour | 30 | your own character is fully black |
| Metroid Prime 4: Beyond | 60 | until it crashes early on |
| Kirby Air Riders | | crashes while loading a race |
| Fire Emblem: Fortune's Weave | 30 | in the first fight, large graphics glitches top left |
| Fast Fusion | | stuck loading into a race, then the emulator crashes |
| Drag x Drive | | no controller support, not playable |

## Download

Releases are on the [releases page](https://github.com/MythosOfs/Orchid/releases). Each one has:

| File | For |
|---|---|
| `orchid-<version>-linux-x86_64-pgo.AppImage` | Linux, profile-guided build (preferred) |
| `orchid-<version>-linux-x86_64.AppImage` | Linux |
| `orchid-<version>-windows-x86_64-pgo.zip` | Windows, profile-guided MinGW build (preferred) |
| `orchid-<version>-windows-x86_64.zip` | Windows |
| `orchid-<version>-macos-arm64.zip` | macOS 15+ on Apple Silicon, experimental |

Linux: `chmod +x` the AppImage and run it; it needs glibc 2.39 or newer. Windows: extract the zip
and run `orchid.exe`.

## Build the newest commit yourself

Linux. Install the dependencies once:

```sh
# Arch / CachyOS
sudo pacman -Syu --needed base-devel boost catch2 cmake enet ffmpeg fmt git glslang libzip lz4 \
  ninja nlohmann-json openssl opus qt6-base qt6-multimedia qt6-charts qt6-wayland sdl3 zlib zstd \
  zip unzip vulkan-headers vulkan-utility-libraries libusb spirv-tools spirv-headers
```

Ubuntu 26.04, Debian 13 and other distributions: see [docs/Deps.md](docs/Deps.md).

Clone `main`, configure and build:

```sh
git clone https://github.com/MythosOfs/Orchid.git
cd Orchid
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DYUZU_TESTS=OFF \
  -DVulkanHeaders_FORCE_BUNDLED=ON -DVulkanUtilityLibraries_FORCE_BUNDLED=ON
cmake --build build
./build/bin/orchid
```

Later, to move to the newest commit (only changed files get rebuilt):

```sh
git pull
cmake --build build
```

The first configure downloads further dependencies into `.cache/cpm`; the two Vulkan flags use the
bundled headers, as the release builds do, so a newer system Vulkan package cannot break the build. A self-built binary has no
update checker; `git pull` is the update. Windows (MSVC) and macOS builds are described in the
[build guide](docs/Build.md).

## Documentation

- [User handbook](docs/user/README.md)
- [Build guide](docs/Build.md) and [dependencies](docs/Deps.md)
- [Fire Emblem: Fortune's Weave bring-up notes](docs/FE_FORTUNES_WEAVE.md)
- [Switch 2 loading notes](NX2_NOTES.md)
- [Development tools](tools)
- [Fork scan: taking fixes from Eglantine forks](docs/ForkScan.md)

## Credits

Orchid takes fixes from Eglantine and its forks; after every release
[tools/fork-scan.sh](docs/ForkScan.md) lists what they fixed since. Each ported commit names its
source and author in its message.

- [zortos293/Eglantine](https://github.com/zortos293/Eglantine) (Zortos), the project Orchid
  started from: paged scalar guest memory access, SPSCQueue value lifetime, timed Event waits on
  Windows, GPU teardown order, buffer cache and query fixes, native Ampere nanosecond timestamps,
  the separate pipeline cache file names.
  Radu Ursache: no mesh pipelines without `VK_EXT_mesh_shader`.
  Kaj (co-authored with Claude Opus 5.5): skip guest uploads for new render targets fully
  overwritten by a clear, adapted with framebuffer render-area, transformed-scissor,
  linear-attachment and zero-layer guards from
  [27226d2568c99c70cf78aa37adaf98b60457b2e5](https://github.com/zortos293/Eglantine/commit/27226d2568c99c70cf78aa37adaf98b60457b2e5).
  Paulo Tumabing: 32-bit global `ATOM.CAS`, adapted to Orchid's storage lowering and atomic routing.
- [Djihads80/Eglantine](https://github.com/Djihads80/Eglantine) and its contributors: merged
  regularly. Idle staging buffers freed (Takoroka), the RangeSet empty-query guard (néo Luigi),
  test fixes (kraibse).
- [ZnOw01/Primrose](https://github.com/ZnOw01/Primrose) (ZnOw01): shader cache invalidation
  lifetime fix, adapted to renderer-thread retirement.
- [exploitizm/Eglantine](https://github.com/exploitizm/Eglantine) (exploitizm): signed IADD32
  negative-overflow boundary fix for GLSL and SPIR-V, adapted from
  [c2f9811073bfe617524d4413742a4d4406fd287a](https://github.com/exploitizm/Eglantine/commit/c2f9811073bfe617524d4413742a4d4406fd287a).
- [Selevet Eglantine](https://github.com/sylvieistaken/selevet-eglantine) (sylvieistaken): the
  windowed GPU mapping table diff, hot loop shortcuts, F2I, local spill and texture query lowering,
  device memory rebuild on layout changes, decoder end of stream, Windows commit retries, and the
  GPU wait knobs (`EDEN_NX2_GPU_WAIT_SPIN_US`, `EDEN_NX2_BACK_SPIN`, `EDEN_NX2_ACQUIRE_QUEUED`,
  `EDEN_NX2_SWAP_STABLE`, `EDEN_NX2_AVX512`).
- [PokeDinner](https://git.eden-emu.dev/Franseven/PokeDinner-a-Eglantine-Fork) (Franseven,
  [mirror](https://github.com/Apollo422/PokeDinner)), partly from Eden-NS2: any cache hint on
  native LDG, SUST with a partial channel mask, `EDEN_AUTO_BUTTONS`.
- Forks of those: Dee1, Greencus and exploitizm (audio manager lifetime, macro binds past the
  start table), LucyferDev (A64 saturating shift and multiply-accumulate instructions), FZout (page
  table block ids, game list titles), sagaopc (NUP directory cycles), Quixomatic (eden-cli, Switch 2
  module names), YLTai (SDL motion after reconnect), Aidan Kinzett and oledz-i (NUP detection,
  geometry stream 0), NicholasBly and ominixisboss (dynarmic), kipavy (storage CAS loop),
  adamhahnHU and arisqz (sparse layered images), Pavel Barabanov and andrewleeog (F2I bits, host1x
  syncpoint actions), MFS-code (Switch 2 GOB order), CarlosKeese and Dennys Kaluzny Pereira
  (lowering), failaip12 (memory manager bounds).
- After 0.6.3: godpow and its contributors nowhere02, xDarkWindx, Duck912, ZnOw01 and CloudP4bbs
  (buffer, texture and staging cache guards, descriptor growth, GPU split wake-up recovery, NPad
  update filter), failaip12 via cira0100 (data races, UBSan fixes, audio and SDL locking),
  monteger8 for adamd (Donkey Kong Bananza buffer cache skips, indirect draw reset by hash), Aidan
  Kinzett (NV12 video decoder), Jonas via jojo-chaechae (x64 call reach, Switch 2 channel setup),
  SavageCore (mod root guards), cblock85, christianball123, Khobzpro, larryliu985, Doughston
  (n-popescu) and Xnos666.
- After 0.7.0: Zortos (native BRA.U, register-handle image access, uniform shifts, global memory
  helpers, the H.264 decoder reset, Aftermath dumps, Switch 2 mixer stream audio), zirulo via Zortos
  (the Ocarina of Time boot fix), Paulo Rossatto (loose base dumps, GPU-only regions, 2D blit
  formats), Raikaru (ssl:s contexts and socket timeouts), Mcm190 (H.264 packet framing), Yuri Olive
  (decoder end of stream), rdenno (fence download ranges), kraibse (tessellation pipelines), a55uka
  (contentless NUP sections), Andreas Heider (ANV barriers), Aidan Kinzett (button glyphs), Jonas
  (H0_NH1 halves), exploitizm (IMAD, I2F and FSET lowering, memory manager reference counts),
  christianball123 (device index fallback), kdegonia-star (global atomic add) and Takoroka (ISETP.EX
  and UIMAD.WIDE lowering).
- rdenno: the Mario Tennis Fever fixes in 0.5.1.
- [Drizz1le/Lichii](https://github.com/Drizz1le/Lichii) (Inc0rporate, with Claude Opus 5.5):
  count nested shader returns before reordering demotes for AMD drivers.
- [DrSergioRFilho/Lichii](https://github.com/DrSergioRFilho/Lichii) (Dr. Sergio, with Claude
  Fable 5.1): apply LayeredFS mods to NUP titles without reapplying their merged update, from
  [fe29f5c45ef8f8ab826f148827a1b3962c32d7d7](https://github.com/DrSergioRFilho/Lichii/commit/fe29f5c45ef8f8ab826f148827a1b3962c32d7d7).
- [chuppatricky/Lichii](https://github.com/chuppatricky/Lichii) (chuppatricky, with Claude Opus 5.5):
  shadow Ampere RED and SURED data registers, adapted from
  [fc5d6dc31b49ae614d06c05302f4ee2abdf8ce4c](https://github.com/chuppatricky/Lichii/commit/fc5d6dc31b49ae614d06c05302f4ee2abdf8ce4c).
- [SoyMoikas/Eglantine](https://github.com/SoyMoikas/Eglantine) (SoyMoikas): uniform-base
  native Ampere local stores and loads, adapted to Orchid's existing vector lowering and register
  shadowing from
  [3c30f286d9631581ea3611c0f30eae46e7f60694](https://github.com/SoyMoikas/Eglantine/commit/3c30f286d9631581ea3611c0f30eae46e7f60694)
  and
  [8c06623a4651e5f31d7297932f6b9887dfea9006](https://github.com/SoyMoikas/Eglantine/commit/8c06623a4651e5f31d7297932f6b9887dfea9006).
- [Hydro74000/Edlws](https://github.com/Hydro74000/Edlws) (Hydro74000): make `UniqueFunction`
  own copies of lvalue callables while moving rvalues, from
  [dd8039f6afc92803938d641a484312931acae9ca](https://github.com/Hydro74000/Edlws/commit/dd8039f6afc92803938d641a484312931acae9ca).
- [DrSergioRFilho/Lichii](https://github.com/DrSergioRFilho/Lichii) (Dr. Sergio,
  with Claude Fable 5.1): check nested config array bounds against the innermost array, from
  [08d3388f0f5496be67dd5f83dfdcddd0fcd59358](https://github.com/DrSergioRFilho/Lichii/commit/08d3388f0f5496be67dd5f83dfdcddd0fcd59358).
- [Shenopi/Lichii](https://github.com/Shenopi/Lichii) (999sian): preserve active custom settings
  while reloading global configuration, adapted from the config-only changes in
  [88ffd1f312f9a38fd45575dc8e1c7d66ef9b4f6e](https://github.com/Shenopi/Lichii/commit/88ffd1f312f9a38fd45575dc8e1c7d66ef9b4f6e).
- [trianglebraindev/Lichii](https://github.com/trianglebraindev/Lichii) (trianglebraindev,
  with Claude Opus 5.5): two-source `F2IP.U8.F32` byte packing, adapted to Orchid's register
  shadowing from
  [ac82c78b5ca4cfbcb11e3043f05a3354f444f027](https://github.com/trianglebraindev/Lichii/commit/ac82c78b5ca4cfbcb11e3043f05a3354f444f027).

- [leadaenima/Lichii](https://github.com/leadaenima/Lichii) (Cursor Agent, co-authored with
  leadaenima): AMDVLK dynamic subgroup shuffle workaround, adapted from
  [957913cf14ca1e3776a4984151b63434af543217](https://github.com/leadaenima/Lichii/commit/957913cf14ca1e3776a4984151b63434af543217)
  and [201450e90a87b494a8d1dc34d32558033a715438](https://github.com/leadaenima/Lichii/commit/201450e90a87b494a8d1dc34d32558033a715438).
  Graphics use the 32-lane broadcast fallback only when subgroup size 32 can be enforced for
  that stage. Compute and unsupported graphics stages retain Shuffle; Qualcomm keeps its
  existing full-subgroup fallback. Pipeline hash diagnostics are omitted.

## License

GPLv3 or any later version, see [LICENSE.txt](LICENSE.txt). Started from
[Eglantine](https://github.com/zortos293/Eglantine), which builds on
[Eden](https://git.eden-emu.dev/eden-emu/eden), yuzu and Sudachi.
