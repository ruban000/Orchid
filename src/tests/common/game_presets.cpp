// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <memory>
#include <catch2/catch_test_macros.hpp>
#include "common/game_presets.h"

namespace {
void SetConflictingPreferences(Settings::Values& settings, bool per_game) {
    const auto set = [per_game](auto& setting, auto value) {
        setting.SetGlobal(!per_game);
        setting.SetValue(value);
    };
    set(settings.renderer_backend, Settings::RendererBackend::Null);
    set(settings.resolution_setup, Settings::ResolutionSetup::Res2X);
    set(settings.gpu_accuracy, Settings::GpuAccuracy::High);
    set(settings.use_asynchronous_shaders, false);
    set(settings.astc_recompression, Settings::AstcRecompression::Uncompressed);
    set(settings.vram_usage_mode, Settings::VramUsageMode::Aggressive);
    set(settings.use_reactive_flushing, true);
    set(settings.vsync_mode, Settings::VSyncMode::Immediate);
    set(settings.scaling_filter, Settings::ScalingFilter::NearestNeighbor);
    set(settings.anti_aliasing, Settings::AntiAliasing::Smaa);
}

void CheckPreset(const Settings::Values& settings) {
    CHECK(settings.renderer_backend.GetValue() == Settings::RendererBackend::Vulkan);
    CHECK(settings.resolution_setup.GetValue() == Settings::ResolutionSetup::Res1X);
    CHECK(settings.gpu_accuracy.GetValue() == Settings::GpuAccuracy::Low);
    CHECK(settings.use_asynchronous_shaders.GetValue());
    CHECK(settings.astc_recompression.GetValue() == Settings::AstcRecompression::Bc3);
    CHECK(settings.vram_usage_mode.GetValue() == Settings::VramUsageMode::Conservative);
    CHECK_FALSE(settings.use_reactive_flushing.GetValue());
    CHECK(settings.vsync_mode.GetValue() == Settings::VSyncMode::FifoRelaxed);
    CHECK(settings.gpu_fence_behavior.GetValue() == Settings::GpuFenceBehavior::Balanced);
    CHECK(settings.async_presentation.GetValue());
    CHECK(settings.scaling_filter.GetValue() == Settings::ScalingFilter::ScaleForce);
    CHECK(settings.anti_aliasing.GetValue() == Settings::AntiAliasing::Fxaa);
}

void CheckPreferences(const Settings::Values& settings, bool global = false) {
    CHECK(settings.renderer_backend.GetValue(global) == Settings::RendererBackend::Null);
    CHECK(settings.resolution_setup.GetValue(global) == Settings::ResolutionSetup::Res2X);
    CHECK(settings.gpu_accuracy.GetValue(global) == Settings::GpuAccuracy::High);
    CHECK_FALSE(settings.use_asynchronous_shaders.GetValue(global));
    CHECK(settings.astc_recompression.GetValue(global) ==
          Settings::AstcRecompression::Uncompressed);
    CHECK(settings.vram_usage_mode.GetValue(global) == Settings::VramUsageMode::Aggressive);
    CHECK(settings.use_reactive_flushing.GetValue(global));
    CHECK(settings.vsync_mode.GetValue(global) == Settings::VSyncMode::Immediate);
    CHECK(settings.scaling_filter.GetValue(global) == Settings::ScalingFilter::NearestNeighbor);
    CHECK(settings.anti_aliasing.GetValue(global) == Settings::AntiAliasing::Smaa);
}
} // namespace

TEST_CASE("Display filter defaults use ScaleForce and FXAA", "[game_presets]") {
    auto settings = std::make_unique<Settings::Values>();
    CHECK(settings->scaling_filter.GetDefault() == Settings::ScalingFilter::ScaleForce);
    CHECK(settings->anti_aliasing.GetDefault() == Settings::AntiAliasing::Fxaa);
}

TEST_CASE("Accepted MTF settings replace visible global preferences", "[game_presets]") {
    auto settings = std::make_unique<Settings::Values>();
    SetConflictingPreferences(*settings, false);
    Settings::SetMarioTennisFeverRecommendedSettings(*settings, true);
    CheckPreset(*settings);
    CHECK(settings->renderer_backend.UsingGlobal());
    CHECK(settings->gpu_accuracy.UsingGlobal());
    CHECK(settings->renderer_backend.GetValue(true) == Settings::RendererBackend::Vulkan);
    CHECK(settings->gpu_accuracy.GetValue(true) == Settings::GpuAccuracy::Low);
}

TEST_CASE("MTF preset overrides saved settings for the session", "[game_presets]") {
    auto settings = std::make_unique<Settings::Values>();
    SetConflictingPreferences(*settings, false);
    SetConflictingPreferences(*settings, true);
    REQUIRE(Settings::ApplyGamePreset(*settings, Settings::MarioTennisFeverProgramId, true));
    CheckPreset(*settings);
    CheckPreferences(*settings, true);
    for (const auto& restore : settings->linkage.restore_functions) {
        restore();
    }
    CheckPreferences(*settings);
    CHECK(settings->renderer_backend.UsingGlobal());
    CHECK(settings->resolution_setup.UsingGlobal());
    REQUIRE(Settings::ApplyGamePreset(*settings, Settings::MarioTennisFeverProgramId, true));
    CheckPreset(*settings);
}

TEST_CASE("MTF preset does not change unrelated games", "[game_presets]") {
    auto settings = std::make_unique<Settings::Values>();
    SetConflictingPreferences(*settings, false);
    CHECK_FALSE(Settings::ApplyGamePreset(*settings, 0, true));
    CHECK_FALSE(
        Settings::ApplyGamePreset(*settings, Settings::MarioTennisFeverProgramId + 1, true));
    CheckPreferences(*settings);
    CHECK(settings->renderer_backend.UsingGlobal());
    CHECK(settings->resolution_setup.UsingGlobal());
}

TEST_CASE("Declining the MTF preset preserves global and per-game preferences", "[game_presets]") {
    auto settings = std::make_unique<Settings::Values>();
    for (bool per_game : {false, true}) {
        SetConflictingPreferences(*settings, per_game);
        CHECK_FALSE(
            Settings::ApplyGamePreset(*settings, Settings::MarioTennisFeverProgramId, false));
        CheckPreferences(*settings);
        CHECK(settings->renderer_backend.UsingGlobal() == !per_game);
        CHECK(settings->resolution_setup.UsingGlobal() == !per_game);
    }
}

TEST_CASE("Every registered preset applies to its own title only", "[game_presets]") {
    for (const auto& preset : Settings::GamePresets) {
        auto settings = std::make_unique<Settings::Values>();
        SetConflictingPreferences(*settings, false);
        CHECK_FALSE(Settings::ApplyGamePreset(*settings, preset.program_id, false));
        CheckPreferences(*settings);
        REQUIRE(Settings::ApplyGamePreset(*settings, preset.program_id, true));
        CHECK(settings->renderer_backend.GetValue() == Settings::RendererBackend::Vulkan);
        CHECK(settings->resolution_setup.GetValue() == Settings::ResolutionSetup::Res1X);
        const bool legacy = preset.program_id == Settings::SuperMarioBrosWonderProgramId ||
                            preset.program_id == Settings::KirbyForgottenLandProgramId;
        CHECK(settings->gpu_accuracy.GetValue() ==
              (legacy ? Settings::GpuAccuracy::High : Settings::GpuAccuracy::Low));
        CHECK(settings->use_reactive_flushing.GetValue() == legacy);
        CheckPreferences(*settings, true);
        CHECK(Settings::FindGamePreset(preset.program_id) == &preset);
    }
    CHECK(Settings::FindGamePreset(0) == nullptr);
}

TEST_CASE("Title presets set what the titles need", "[game_presets]") {
    auto settings = std::make_unique<Settings::Values>();
    REQUIRE(Settings::ApplyGamePreset(*settings, Settings::DonkeyKongBananzaProgramId, true));
    CHECK(settings->gpu_clock.GetValue() == Settings::GpuClock::Boost);
    CHECK(settings->fps30_patch.GetValue());
    settings = std::make_unique<Settings::Values>();
    REQUIRE(Settings::ApplyGamePreset(*settings, Settings::HyruleWarriorsAgeOfImprisonmentProgramId,
                                      true));
    CHECK(settings->cpu_clock.GetValue() == Settings::CpuClock::Normal);
    CHECK(settings->gpu_fence_behavior.GetValue() == Settings::GpuFenceBehavior::Balanced);
    settings = std::make_unique<Settings::Values>();
    REQUIRE(Settings::ApplyGamePreset(*settings, Settings::MarioTennisFeverProgramId, true));
    CHECK(settings->gpu_fence_behavior.GetValue() == Settings::GpuFenceBehavior::Balanced);
    CHECK(settings->async_presentation.GetValue());
}

TEST_CASE("Graphics quality tokens are stable and legacy environment answers remain deterministic",
          "[game_presets]") {
    using Quality = Settings::GamePresetQuality;
    for (const auto quality : {Quality::Low, Quality::Standard, Quality::High}) {
        const auto token = Settings::GamePresetQualityToken(quality);
        REQUIRE(Settings::ParseGamePresetQuality(token) == quality);
        CHECK(Settings::GamePresetEnvironmentQuality(token) == quality);
    }
    CHECK(Settings::GamePresetEnvironmentQuality("1") == Quality::Standard);
    CHECK_FALSE(Settings::GamePresetEnvironmentQuality("0"));
    for (const auto token : {"", "2", "HIGH", "unknown", " low"}) {
        CHECK_FALSE(Settings::ParseGamePresetQuality(token));
        CHECK_FALSE(Settings::GamePresetEnvironmentQuality(token));
    }
}

TEST_CASE("Every visual quality retains each title's compatibility settings and saved globals",
          "[game_presets]") {
    using Quality = Settings::GamePresetQuality;
    for (const auto& preset : Settings::GamePresets) {
        for (const auto quality : {Quality::Low, Quality::Standard, Quality::High}) {
            auto settings = std::make_unique<Settings::Values>();
            SetConflictingPreferences(*settings, false);
            SetConflictingPreferences(*settings, true);
            settings->max_anisotropy.SetGlobal(true);
            settings->max_anisotropy.SetValue(Settings::AnisotropyMode::X8);
            REQUIRE(Settings::ApplyGamePreset(*settings, preset.program_id, quality));
            CHECK(settings->resolution_setup.GetValue() ==
                  (quality == Quality::Low    ? Settings::ResolutionSetup::Res3_4X
                   : quality == Quality::High ? Settings::ResolutionSetup::Res2X
                                              : Settings::ResolutionSetup::Res1X));
            CHECK(settings->anti_aliasing.GetValue() == (quality == Quality::High
                                                             ? Settings::AntiAliasing::Smaa
                                                             : Settings::AntiAliasing::Fxaa));
            CHECK(settings->astc_recompression.GetValue() ==
                  (quality == Quality::High ? Settings::AstcRecompression::Uncompressed
                                            : Settings::AstcRecompression::Bc3));
            CHECK(settings->max_anisotropy.GetValue() ==
                  (quality == Quality::Low    ? Settings::AnisotropyMode::X2
                   : quality == Quality::High ? Settings::AnisotropyMode::X16
                                              : Settings::AnisotropyMode::X4));
            CHECK(settings->renderer_backend.GetValue() == Settings::RendererBackend::Vulkan);
            const bool legacy = preset.program_id == Settings::SuperMarioBrosWonderProgramId ||
                                preset.program_id == Settings::KirbyForgottenLandProgramId;
            CHECK(settings->gpu_accuracy.GetValue() ==
                  (legacy ? Settings::GpuAccuracy::High : Settings::GpuAccuracy::Low));
            CHECK(settings->use_asynchronous_shaders.GetValue());
            CHECK(settings->vram_usage_mode.GetValue() == Settings::VramUsageMode::Conservative);
            CHECK(settings->use_reactive_flushing.GetValue() == legacy);
            CHECK(settings->vsync_mode.GetValue() == Settings::VSyncMode::FifoRelaxed);
            CHECK(settings->scaling_filter.GetValue() == Settings::ScalingFilter::ScaleForce);
            switch (preset.program_id) {
            case Settings::DonkeyKongBananzaProgramId:
                CHECK(settings->gpu_clock.GetValue() == Settings::GpuClock::Boost);
                CHECK(settings->fps30_patch.GetValue());
                break;
            case Settings::HyruleWarriorsAgeOfImprisonmentProgramId:
                CHECK(settings->cpu_clock.GetValue() == Settings::CpuClock::Normal);
                CHECK(settings->gpu_fence_behavior.GetValue() ==
                      Settings::GpuFenceBehavior::Balanced);
                break;
            case Settings::MarioTennisFeverProgramId:
                CHECK(settings->gpu_fence_behavior.GetValue() ==
                      Settings::GpuFenceBehavior::Balanced);
                CHECK(settings->async_presentation.GetValue());
                break;
            }
            CheckPreferences(*settings, true);
            CHECK(settings->max_anisotropy.GetValue(true) == Settings::AnisotropyMode::X8);
            for (const auto& restore : settings->linkage.restore_functions) {
                restore();
            }
            CheckPreferences(*settings);
            CHECK(settings->max_anisotropy.GetValue() == Settings::AnisotropyMode::X8);
        }
    }
}

TEST_CASE("Changing quality replaces every visual override and ignores unsupported titles",
          "[game_presets]") {
    using Quality = Settings::GamePresetQuality;
    auto settings = std::make_unique<Settings::Values>();
    const u64 title = Settings::MarioTennisFeverProgramId;
    REQUIRE(Settings::ApplyGamePreset(*settings, title, Quality::High));
    REQUIRE(Settings::ApplyGamePreset(*settings, title, Quality::Low));
    CHECK(settings->resolution_setup.GetValue() == Settings::ResolutionSetup::Res3_4X);
    CHECK(settings->anti_aliasing.GetValue() == Settings::AntiAliasing::Fxaa);
    CHECK(settings->astc_recompression.GetValue() == Settings::AstcRecompression::Bc3);
    CHECK(settings->max_anisotropy.GetValue() == Settings::AnisotropyMode::X2);
    REQUIRE(Settings::ApplyGamePreset(*settings, title, true));
    CHECK(settings->resolution_setup.GetValue() == Settings::ResolutionSetup::Res1X);
    CHECK(settings->max_anisotropy.GetValue() == Settings::AnisotropyMode::X4);
    CHECK_FALSE(Settings::ApplyGamePreset(*settings, 0, Quality::High));
    CHECK_FALSE(Settings::ApplyGamePreset(*settings, Settings::MarioTennisFeverProgramId + 1,
                                          Quality::Low));
    CHECK(settings->resolution_setup.GetValue() == Settings::ResolutionSetup::Res1X);
}

TEST_CASE("The profile registry includes exactly the nine identified owned games",
          "[game_presets]") {
    constexpr std::array expected{
        Settings::MarioTennisFeverProgramId,    Settings::HyruleWarriorsAgeOfImprisonmentProgramId,
        Settings::DonkeyKongBananzaProgramId,   Settings::KirbyAirRidersProgramId,
        Settings::PokemonPokopiaProgramId,      Settings::StarFoxProgramId,
        Settings::YoshiMysteriousBookProgramId, Settings::SuperMarioBrosWonderProgramId,
        Settings::KirbyForgottenLandProgramId};
    REQUIRE(std::size(Settings::GamePresets) == expected.size());
    for (const u64 id : expected) {
        const auto* preset = Settings::FindGamePreset(id);
        REQUIRE(preset != nullptr);
        CHECK(preset->program_id == id);
    }
}

TEST_CASE("Out-of-scope and different-edition game IDs do not receive graphics profiles",
          "[game_presets]") {
    constexpr std::array excluded{Settings::MarioKartWorldProgramId,
                                  u64{0x0400486024DEC000},          // Ocarina of Time
                                  u64{0x04007C6028AA8000},          // Other Yoshi dump, not Full
                                  u64{0}, u64{0x040074A01BF13001}}; // Dummy ID and Bananza add-on
    for (const u64 id : excluded) {
        CAPTURE(id);
        auto settings = std::make_unique<Settings::Values>();
        SetConflictingPreferences(*settings, false);
        SetConflictingPreferences(*settings, true);
        CHECK(Settings::FindGamePreset(id) == nullptr);
        for (const auto quality :
             {Settings::GamePresetQuality::Low, Settings::GamePresetQuality::Standard,
              Settings::GamePresetQuality::High}) {
            CHECK_FALSE(Settings::ApplyGamePreset(*settings, id, quality));
            CheckPreferences(*settings);
            CheckPreferences(*settings, true);
            CHECK_FALSE(settings->renderer_backend.UsingGlobal());
        }
    }
}

TEST_CASE("Wonder and Forgotten Land presets preserve legacy synchronization and clock choices",
          "[game_presets]") {
    for (const u64 id :
         {Settings::SuperMarioBrosWonderProgramId, Settings::KirbyForgottenLandProgramId}) {
        CAPTURE(id);
        for (const bool per_game : {false, true}) {
            auto settings = std::make_unique<Settings::Values>();
            SetConflictingPreferences(*settings, per_game);
            const auto set = [per_game](auto& setting, auto value) {
                setting.SetGlobal(!per_game);
                setting.SetValue(value);
            };
            set(settings->gpu_fence_behavior, Settings::GpuFenceBehavior::Accurate);
            set(settings->cpu_clock, Settings::CpuClock::Overclock);
            set(settings->gpu_clock, Settings::GpuClock::Overclock);
            set(settings->async_presentation, false);
            set(settings->fps30_patch, false);
            set(settings->use_docked_mode, Settings::ConsoleMode::Docked);
            for (const auto quality :
                 {Settings::GamePresetQuality::Low, Settings::GamePresetQuality::Standard,
                  Settings::GamePresetQuality::High}) {
                REQUIRE(Settings::ApplyGamePreset(*settings, id, quality));
                CHECK(settings->gpu_accuracy.GetValue() == Settings::GpuAccuracy::High);
                CHECK(settings->use_reactive_flushing.GetValue());
                CHECK(settings->gpu_fence_behavior.GetValue() ==
                      Settings::GpuFenceBehavior::Accurate);
                CHECK(settings->cpu_clock.GetValue() == Settings::CpuClock::Overclock);
                CHECK(settings->gpu_clock.GetValue() == Settings::GpuClock::Overclock);
                CHECK_FALSE(settings->async_presentation.GetValue());
                CHECK_FALSE(settings->fps30_patch.GetValue());
                CHECK(settings->use_docked_mode.GetValue() == Settings::ConsoleMode::Docked);
                CHECK(settings->gpu_accuracy.UsingGlobal() == !per_game);
                CHECK(settings->use_reactive_flushing.UsingGlobal() == !per_game);
                CHECK(settings->gpu_fence_behavior.UsingGlobal() == !per_game);
                CHECK(settings->cpu_clock.UsingGlobal() == !per_game);
                CHECK(settings->gpu_clock.UsingGlobal() == !per_game);
                CHECK(settings->async_presentation.UsingGlobal() == !per_game);
                CHECK(settings->fps30_patch.UsingGlobal() == !per_game);
                CHECK(settings->use_docked_mode.UsingGlobal() == !per_game);
            }
        }
    }
}

TEST_CASE("New native profiles do not invent clock, fence, frame-cap or presentation overrides",
          "[game_presets]") {
    constexpr std::array ids{Settings::KirbyAirRidersProgramId, Settings::PokemonPokopiaProgramId,
                             Settings::StarFoxProgramId, Settings::YoshiMysteriousBookProgramId};
    for (const u64 id : ids) {
        auto settings = std::make_unique<Settings::Values>();
        const auto set = [&](auto& setting, auto value) {
            setting.SetGlobal(false);
            setting.SetValue(value);
        };
        set(settings->gpu_fence_behavior, Settings::GpuFenceBehavior::Accurate);
        set(settings->cpu_clock, Settings::CpuClock::Overclock);
        set(settings->gpu_clock, Settings::GpuClock::Overclock);
        set(settings->async_presentation, false);
        set(settings->fps30_patch, false);
        set(settings->use_docked_mode, Settings::ConsoleMode::Docked);
        for (const auto quality :
             {Settings::GamePresetQuality::Low, Settings::GamePresetQuality::Standard,
              Settings::GamePresetQuality::High}) {
            REQUIRE(Settings::ApplyGamePreset(*settings, id, quality));
            CHECK(settings->gpu_fence_behavior.GetValue() == Settings::GpuFenceBehavior::Accurate);
            CHECK(settings->cpu_clock.GetValue() == Settings::CpuClock::Overclock);
            CHECK(settings->gpu_clock.GetValue() == Settings::GpuClock::Overclock);
            CHECK_FALSE(settings->async_presentation.GetValue());
            CHECK_FALSE(settings->fps30_patch.GetValue());
            CHECK(settings->use_docked_mode.GetValue() == Settings::ConsoleMode::Docked);
            CHECK_FALSE(settings->gpu_fence_behavior.UsingGlobal());
            CHECK_FALSE(settings->cpu_clock.UsingGlobal());
            CHECK_FALSE(settings->gpu_clock.UsingGlobal());
        }
    }
}
