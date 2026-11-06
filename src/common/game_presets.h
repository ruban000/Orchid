// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <optional>
#include <string_view>

#include "common/settings.h"

namespace Settings {

enum class GamePresetQuality { Low, Standard, High };

inline constexpr std::string_view GamePresetQualityToken(GamePresetQuality quality) {
    switch (quality) {
    case GamePresetQuality::Low:
        return "low";
    case GamePresetQuality::High:
        return "high";
    default:
        return "standard";
    }
}

inline constexpr std::optional<GamePresetQuality> ParseGamePresetQuality(std::string_view token) {
    if (token == "low")
        return GamePresetQuality::Low;
    if (token == "standard")
        return GamePresetQuality::Standard;
    if (token == "high")
        return GamePresetQuality::High;
    return std::nullopt;
}

// Legacy headless answers retain their meaning; quality tokens also bypass the dialog.
// Environment answers apply to this boot only and never become a saved UI preference.
inline constexpr std::optional<GamePresetQuality> GamePresetEnvironmentQuality(
    std::string_view value) {
    return value == "1" ? std::optional{GamePresetQuality::Standard}
                        : ParseGamePresetQuality(value);
}

inline constexpr u64 MarioKartWorldProgramId = 0x0400C3F00006E000ULL;
inline constexpr u64 DonkeyKongBananzaProgramId = 0x040074A01BF12000ULL;
inline constexpr u64 HyruleWarriorsAgeOfImprisonmentProgramId = 0x04003F40232DE000ULL;
inline constexpr u64 MarioTennisFeverProgramId = 0x0400D4201C6EA000ULL;
// Confirmed from the local programs' NUP/NPDM metadata, including the full Yoshi release.
inline constexpr u64 KirbyAirRidersProgramId = 0x040077201BE50000ULL;
inline constexpr u64 PokemonPokopiaProgramId = 0x0400A3A01EE4C000ULL;
inline constexpr u64 StarFoxProgramId = 0x0400E88025B1A000ULL;
inline constexpr u64 YoshiMysteriousBookProgramId = 0x0400245021D2C000ULL;
inline constexpr u64 SuperMarioBrosWonderProgramId = 0x010015100B514000ULL;
inline constexpr u64 KirbyForgottenLandProgramId = 0x01004D300C5AE000ULL;

namespace Detail {
inline auto SessionSetter(bool global) {
    return [global](auto& setting, auto value) {
        setting.SetGlobal(global);
        setting.SetValue(value);
    };
}

/// Shared graphics baseline. Do not impose native GPU synchronization or clock settings here:
/// Programs with the Switch 1 layout retain those preferences.
inline void SetGraphicsBaseSettings(Values& settings, bool global) {
    const auto session = SessionSetter(global);
    session(settings.renderer_backend, RendererBackend::Vulkan);
    session(settings.resolution_setup, ResolutionSetup::Res1X);
    session(settings.use_asynchronous_shaders, true);
    session(settings.astc_recompression, AstcRecompression::Bc3);
    session(settings.vram_usage_mode, VramUsageMode::Conservative);
    session(settings.vsync_mode, VSyncMode::FifoRelaxed);
    session(settings.scaling_filter, ScalingFilter::ScaleForce);
    session(settings.anti_aliasing, AntiAliasing::Fxaa);
}

inline void SetSwitch2BaseSettings(Values& settings, bool global) {
    SetGraphicsBaseSettings(settings, global);
    const auto session = SessionSetter(global);
    session(settings.gpu_accuracy, GpuAccuracy::Low);
    session(settings.use_reactive_flushing, false);
}
} // namespace Detail

inline void SetDonkeyKongBananzaRecommendedSettings(Values& settings, bool global) {
    Detail::SetSwitch2BaseSettings(settings, global);
    const auto session = Detail::SessionSetter(global);
    // The game picks its resolution and swap interval from GPU timestamps (game_settings.cpp).
    session(settings.gpu_clock, GpuClock::Boost);
    session(settings.fps30_patch, true);
}

inline void SetHyruleWarriorsRecommendedSettings(Values& settings, bool global) {
    Detail::SetSwitch2BaseSettings(settings, global);
    const auto session = Detail::SessionSetter(global);
    // A faster CPU clock speeds the game itself up (game_settings.cpp).
    session(settings.cpu_clock, CpuClock::Normal);
    session(settings.gpu_fence_behavior, GpuFenceBehavior::Balanced);
}

inline void SetMarioTennisFeverRecommendedSettings(Values& settings, bool global) {
    Detail::SetSwitch2BaseSettings(settings, global);
    const auto session = Detail::SessionSetter(global);
    // Balanced fences keep the characters visible; presenting from its own thread keeps 60 fps.
    session(settings.gpu_fence_behavior, GpuFenceBehavior::Balanced);
    session(settings.async_presentation, true);
}

struct GamePreset {
    u64 program_id;
    const char* name;
    void (*apply)(Values& settings, bool global);
};

inline constexpr GamePreset GamePresets[] = {
    {MarioTennisFeverProgramId, "Mario Tennis Fever", SetMarioTennisFeverRecommendedSettings},
    {DonkeyKongBananzaProgramId, "Donkey Kong Bananza", SetDonkeyKongBananzaRecommendedSettings},
    {HyruleWarriorsAgeOfImprisonmentProgramId, "Hyrule Warriors: Age of Imprisonment",
     SetHyruleWarriorsRecommendedSettings},
    {KirbyAirRidersProgramId, "Kirby Air Riders", Detail::SetSwitch2BaseSettings},
    {PokemonPokopiaProgramId, "Pokemon Pokopia", Detail::SetSwitch2BaseSettings},
    {StarFoxProgramId, "Star Fox", Detail::SetSwitch2BaseSettings},
    {YoshiMysteriousBookProgramId, "Yoshi and the Mysterious Book", Detail::SetSwitch2BaseSettings},
    {SuperMarioBrosWonderProgramId, "Super Mario Bros. Wonder", Detail::SetGraphicsBaseSettings},
    {KirbyForgottenLandProgramId, "Kirby and the Forgotten Land", Detail::SetGraphicsBaseSettings},
};

/// The title's preset, or nullptr when it has none.
inline const GamePreset* FindGamePreset(u64 program_id) {
    for (const GamePreset& preset : GamePresets) {
        if (preset.program_id == program_id) {
            return &preset;
        }
    }
    return nullptr;
}

/// Applies the title's compatibility settings and visual quality for this session. Global and
/// saved per-game values remain intact; RestoreGlobalState drops these overrides on stop.
inline bool ApplyGamePreset(Values& settings, u64 program_id, GamePresetQuality quality) {
    const GamePreset* const preset = FindGamePreset(program_id);
    if (preset == nullptr) {
        return false;
    }
    preset->apply(settings, false);
    const auto session = Detail::SessionSetter(false);
    switch (quality) {
    case GamePresetQuality::Low:
        session(settings.resolution_setup, ResolutionSetup::Res3_4X);
        session(settings.anti_aliasing, AntiAliasing::Fxaa);
        session(settings.astc_recompression, AstcRecompression::Bc3);
        session(settings.max_anisotropy, AnisotropyMode::X2);
        break;
    case GamePresetQuality::High:
        session(settings.resolution_setup, ResolutionSetup::Res2X);
        session(settings.anti_aliasing, AntiAliasing::Smaa);
        session(settings.astc_recompression, AstcRecompression::Uncompressed);
        session(settings.max_anisotropy, AnisotropyMode::X16);
        break;
    default:
        session(settings.resolution_setup, ResolutionSetup::Res1X);
        session(settings.anti_aliasing, AntiAliasing::Fxaa);
        session(settings.astc_recompression, AstcRecompression::Bc3);
        session(settings.max_anisotropy, AnisotropyMode::X4);
        break;
    }
    return true;
}

inline bool ApplyGamePreset(Values& settings, u64 program_id, bool accepted) {
    return accepted && ApplyGamePreset(settings, program_id, GamePresetQuality::Standard);
}

} // namespace Settings
