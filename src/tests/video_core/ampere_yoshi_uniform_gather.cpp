// SPDX-License-Identifier: GPL-3.0-or-later
// Maxwell TLD4.B lowering based on upstream 563ed8e60b; captured Yoshi words from
// support-games-tests/yoshi/navigation-1/eden_log.txt, not synthetic whole shaders.
#include <array>
#include <catch2/catch_test_macros.hpp>
#include "shader_recompiler/environment.h"
#include "shader_recompiler/frontend/ampere/lower_to_maxwell.h"
#include "shader_recompiler/frontend/ir/basic_block.h"
#include "shader_recompiler/frontend/ir/modifiers.h"
#include "shader_recompiler/frontend/maxwell/translate/impl/impl.h"

namespace {
using namespace Shader;
using Ampere::Instruction;
constexpr Instruction Exit{{0x0000794d, 0, 0x03800000, 0}};
constexpr std::array Captured{
    Instruction{{0x1d0c7f63, 0x30000a1e, 0x081e0f14, 0x000f4200}},
    Instruction{{0x25147f63, 0x30000a26, 0x081e0f1c, 0x000f4200}},
    Instruction{{0x2d0e7f63, 0x30000420, 0x0c1e0f3c, 0x000f4200}},
};
class GatherEnvironment final : public Environment {
public:
    u64 ReadInstruction(u32) override {
        return 0;
    }
    u32 ReadCbufValue(u32, u32) override {
        return 0;
    }
    TextureType ReadTextureType(u32) override {
        return TextureType::Color2D;
    }
    TexturePixelFormat ReadTexturePixelFormat(u32) override {
        return TexturePixelFormat::A8B8G8R8_UNORM;
    }
    bool IsTexturePixelFormatInteger(u32) override {
        return false;
    }
    u32 ReadViewportTransformState() override {
        return 0;
    }
    u32 TextureBoundBuffer() const override {
        return 0;
    }
    u32 LocalMemorySize() const override {
        return 0;
    }
    u32 SharedMemorySize() const override {
        return 0;
    }
    std::array<u32, 3> WorkgroupSize() const override {
        return {1, 1, 1};
    }
    bool HasHLEMacroState() const override {
        return false;
    }
    std::optional<ReplaceConstant> GetReplaceConstBuffer(u32, u32) override {
        return {};
    }
    void Dump(u64, u64) override {}
};


TEST_CASE("Yoshi captured uniform gathers reach Maxwell bindless 2D gather IR",
          "[video_core][yoshi_gather]") {
    constexpr std::array x{29U, 37U, 45U}, y{30U, 38U, 32U}, handle{226U, 226U, 220U};
    constexpr std::array destinations{std::array{12U, 13U, 20U, 21U},
                                      std::array{20U, 21U, 28U, 29U},
                                      std::array{14U, 15U, 60U, 61U}};
    for (size_t index = 0; index < Captured.size(); ++index) {
        CAPTURE(index);
        const auto lowered = Ampere::LowerToMaxwell(std::array{Captured[index], Exit});
        REQUIRE(lowered);
        REQUIRE(lowered.instructions.size() == 9);
        CHECK(lowered.required_texture_buffer == ~0U);
        CHECK(((lowered.instructions[0] >> 20) & 255) == x[index]);
        CHECK(((lowered.instructions[1] >> 20) & 255) == y[index]);
        CHECK(((lowered.instructions[2] >> 20) & 255) == handle[index]);
        for (size_t channel = 0; channel < 4; ++channel) {
            CHECK((lowered.instructions[4 + channel] & 255) == destinations[index][channel]);
            CHECK(((lowered.instructions[4 + channel] >> 20) & 255) == 250 + channel);
        }
        GatherEnvironment env;
        ObjectPool<IR::Inst> pool;
        IR::Block block{pool};
        Maxwell::TranslatorVisitor visitor{env, block};
        // Real Maxwell decoding independently checks the emitted opcode's operand layout.
        visitor.TLD4_b(lowered.instructions[3]);
        unsigned gathers{};
        for (const IR::Inst& instruction : block) {
            if (instruction.GetOpcode() != IR::Opcode::BindlessImageGather) continue;
            ++gathers;
            const auto info = instruction.Flags<IR::TextureInstInfo>();
            CHECK(info.type.Value() == TextureType::Color2D);
            CHECK(info.is_depth.Value() == 0);
            CHECK(info.gather_component.Value() == 0);
            CHECK(instruction.Arg(0).Inst()->GetOpcode() == IR::Opcode::GetRegister);
            CHECK(instruction.Arg(0).Inst()->Arg(0).Reg() == IR::Reg{246});
            CHECK(instruction.Arg(1).Inst()->GetOpcode() == IR::Opcode::CompositeConstructF32x2);
            CHECK(instruction.Arg(2).IsEmpty());
            CHECK(instruction.Arg(3).IsEmpty());
        }
        CHECK(gathers == 1);
    }
}

TEST_CASE("Uniform gathers preserve private UR mappings and component selectors",
          "[video_core][yoshi_gather]") {
    for (const auto [uniform, mirror] : std::array{
             std::pair{0U, 216U}, std::pair{21U, 237U},
             std::pair{22U, 240U}, std::pair{27U, 245U}}) {
        for (u32 component = 0; component < 4; ++component) {
            auto instruction = Captured[0];
            instruction.words[1] = (instruction.words[1] & ~0x3f00U) | (uniform << 8);
            instruction.words[2] = (instruction.words[2] & ~0x01800000U) | (component << 23);
            const auto lowered = Ampere::LowerToMaxwell(std::array{instruction, Exit});
            REQUIRE(lowered);
            CHECK(((lowered.instructions[2] >> 20) & 255) == mirror);
            CHECK(((lowered.instructions[3] >> 38) & 3) == component);
            CHECK(((lowered.instructions[3] >> 36) & 3) == 0); // no offsets
            CHECK(((lowered.instructions[3] >> 50) & 1) == 0); // no depth compare
        }
    }
    // A predicated gather keeps every data movement under that same guest predicate.
    auto guarded = Captured[0];
    guarded.words[0] = (guarded.words[0] & ~0xf000U) | 0x2000U;
    const auto lowered = Ampere::LowerToMaxwell(std::array{guarded, Exit});
    REQUIRE(lowered);
    for (size_t i = 0; i < 8; ++i) CHECK(((lowered.instructions[i] >> 16) & 15) == 2);
}

TEST_CASE("Uniform gathers reject unverified modes and register ranges",
          "[video_core][yoshi_gather]") {
    // Offset, PTP, DC, F16, sparse predicate, array dimensions, reserved texture operands,
    // extended instruction modifiers: none is enabled by this narrow port.
    for (const auto [word, bit] : std::array{
             std::pair{1U, 14U}, std::pair{1U, 27U}, std::pair{1U, 28U},
             std::pair{1U, 29U}, std::pair{1U, 30U}, std::pair{2U, 12U}, std::pair{2U, 13U},
             std::pair{2U, 14U}, std::pair{2U, 15U}, std::pair{2U, 17U},
             std::pair{2U, 25U}, std::pair{2U, 27U}, std::pair{3U, 0U},
             std::pair{3U, 8U}, std::pair{3U, 30U}, std::pair{3U, 31U}}) {
        auto instruction = Captured[0];
        instruction.words[word] ^= 1U << bit;
        CHECK_FALSE(Ampere::LowerToMaxwell(std::array{instruction, Exit}));
    }
    for (const u32 uniform : {28U, 62U, 63U}) {
        auto instruction = Captured[0];
        instruction.words[1] = (instruction.words[1] & ~0x3f00U) | (uniform << 8);
        CHECK_FALSE(Ampere::LowerToMaxwell(std::array{instruction, Exit}));
    }
    auto instruction = Captured[0];
    instruction.words[2] &= ~0xf00U;
    CHECK_FALSE(Ampere::LowerToMaxwell(std::array{instruction, Exit}));
    for (const u32 bit : {16U, 24U, 32U, 64U}) {
        auto boundary = Captured[0];
        // A split two-register destination starting at215 would clobber UR0 at216.
        // Coordinate sources at216 are reserved even if other operands are low.
        const u32 reg = bit == 16 || bit == 64 ? 215 : 216;
        boundary.words[bit / 32] = (boundary.words[bit / 32] & ~(255U << (bit % 32))) |
                                    (reg << (bit % 32));
        CHECK_FALSE(Ampere::LowerToMaxwell(std::array{boundary, Exit}));
    }
}

TEST_CASE("Yoshi captured uniform array gather preserves layer and coordinate order",
          "[video_core][yoshi_gather]") {
    // Next instruction at0x1280 in4422e006ec319f8c-cs-1af4645c00.bin.
    const Instruction captured{{0x2c2e7f63, 0xb0000820, 0x0c1e0f2c, 0x000f4200}};
    const auto lowered = Ampere::LowerToMaxwell(std::array{captured, Exit});
    REQUIRE(lowered);
    REQUIRE(lowered.instructions.size() == 10);
    CHECK(((lowered.instructions[0] >> 20) & 255) == 44); // layer
    CHECK(((lowered.instructions[1] >> 20) & 255) == 45); // x = Ra+1
    CHECK(((lowered.instructions[2] >> 20) & 255) == 32); // y = Rb
    CHECK(((lowered.instructions[3] >> 20) & 255) == 224); // UR8, copied to246
    CHECK((lowered.instructions[3] & 255) == 246);
    GatherEnvironment env;
    ObjectPool<IR::Inst> pool;
    IR::Block block{pool};
    Maxwell::TranslatorVisitor visitor{env, block};
    visitor.TLD4_b(lowered.instructions[4]);
    unsigned gathers{};
    for (const IR::Inst& instruction : block) {
        if (instruction.GetOpcode() != IR::Opcode::BindlessImageGather) continue;
        ++gathers;
        const auto info = instruction.Flags<IR::TextureInstInfo>();
        CHECK(info.type.Value() == TextureType::ColorArray2D);
        CHECK(info.is_depth.Value() == 0);
        CHECK(instruction.Arg(0).Inst()->Arg(0).Reg() == IR::Reg{246});
        const auto* coords = instruction.Arg(1).Inst();
        REQUIRE(coords->GetOpcode() == IR::Opcode::CompositeConstructF32x3);
        const auto origin = [](IR::Value value) {
            for (unsigned depth = 0; depth < 8 && !value.IsImmediate(); ++depth) {
                const auto* inst = value.Inst();
                if (inst->GetOpcode() == IR::Opcode::GetRegister) return inst->Arg(0).Reg();
                value = inst->Arg(0);
            }
            return IR::Reg{255};
        };
        CHECK(origin(coords->Arg(0)) == IR::Reg{251}); // x
        CHECK(origin(coords->Arg(1)) == IR::Reg{252}); // y
        CHECK(origin(coords->Arg(2)) == IR::Reg{250}); // converted u16 layer
        CHECK(instruction.Arg(2).IsEmpty());
        CHECK(instruction.Arg(3).IsEmpty());
    }
    CHECK(gathers == 1);
    constexpr std::array destinations{46U, 47U, 44U, 45U};
    for (size_t i = 0; i < destinations.size(); ++i)
        CHECK((lowered.instructions[5 + i] & 255) == destinations[i]);
    // Both discarded halves and overlapping input/output registers remain well-defined:
    // all coordinates and the uniform handle are staged before any result move.
    for (const bool first_discarded : {false, true}) {
        auto discarded = captured;
        if (first_discarded) discarded.words[0] |= 255U << 16;
        else discarded.words[2] |= 255U;
        REQUIRE(Ampere::LowerToMaxwell(std::array{discarded, Exit}));
    }
}

TEST_CASE("Uniform array gather retains coordinate pair and mode bounds",
          "[video_core][yoshi_gather]") {
    const Instruction captured{{0x2c2e7f63, 0xb0000820, 0x0c1e0f2c, 0x000f4200}};
    for (const u32 ra : {215U, 216U, 254U, 255U}) {
        auto invalid = captured;
        invalid.words[0] = (invalid.words[0] & 0x00ffffffU) | (ra << 24);
        CHECK_FALSE(Ampere::LowerToMaxwell(std::array{invalid, Exit}));
    }
    for (const auto [word, bit] : std::array{
             std::pair{1U, 29U}, std::pair{1U, 30U}, std::pair{2U, 12U},
             std::pair{2U, 13U}, std::pair{2U, 14U}, std::pair{2U, 15U}}) {
        auto invalid = captured;
        invalid.words[word] ^= 1U << bit;
        CHECK_FALSE(Ampere::LowerToMaxwell(std::array{invalid, Exit}));
    }
    for (const u32 bit : {16U, 64U}) {
        auto invalid = captured;
        invalid.words[bit / 32] = (invalid.words[bit / 32] & ~(255U << (bit % 32))) |
                                  (215U << (bit % 32));
        CHECK_FALSE(Ampere::LowerToMaxwell(std::array{invalid, Exit}));
    }
}
} // namespace
