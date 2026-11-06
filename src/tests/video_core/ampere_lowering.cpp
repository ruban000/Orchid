// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <catch2/catch_test_macros.hpp>

#include "shader_recompiler/frontend/ampere/lower_to_maxwell.h"
#include "shader_recompiler/frontend/ampere/program_capture.h"
#include "shader_recompiler/frontend/ir/program.h"
#include "shader_recompiler/frontend/maxwell/translate/translate.h"
#include "shader_recompiler/ir_opt/passes.h"

namespace {
using Shader::Ampere::Instruction;
using Shader::Ampere::LowerToMaxwell;
using Shader::Ampere::ProgramCaptureState;

constexpr Instruction EXIT{{0x0000794d, 0, 0x03800000, 0}};

// Lowered Maxwell words whose top bits (above `shift`) are `opcode`.
template <typename Program>
size_t CountMaxwell(const Program& lowered, u64 opcode, u32 shift) {
    return static_cast<size_t>(std::count_if(lowered.instructions.begin(),
                                             lowered.instructions.end(),
                                             [&](u64 word) { return (word >> shift) == opcode; }));
}

template <typename Program>
u64 FindMaxwell(const Program& lowered, u64 opcode, u32 shift) {
    const auto it = std::find_if(lowered.instructions.begin(), lowered.instructions.end(),
                                 [&](u64 word) { return (word >> shift) == opcode; });
    return it != lowered.instructions.end() ? *it : 0;
}

TEST_CASE("Ampere BRA.CONV always branches and BRA.DIV never does", "[video_core][ampere]") {
    // BRA.CONV UR6 / BRA.DIV ~URZ to the second EXIT: the mode is in the offset's low two bits.
    const std::array<Instruction, 3> convergent{{
        {{0x06007947, 0x10 | 3, 0x0b800000, 0}},
        EXIT,
        EXIT,
    }};
    ProgramCaptureState capture;
    REQUIRE(capture.Observe(convergent[0], 0) == ProgramCaptureState::Result::Continue);
    const auto taken = LowerToMaxwell(convergent);
    REQUIRE(static_cast<bool>(taken));
    REQUIRE(CountMaxwell(taken, 0xe24, 52) == 1);

    const std::array<Instruction, 3> divergent{{
        {{0xff007947, 0x10 | 2, 0x0b800000, 0}},
        EXIT,
        EXIT,
    }};
    const auto dropped = LowerToMaxwell(divergent);
    REQUIRE(static_cast<bool>(dropped));
    REQUIRE(CountMaxwell(dropped, 0xe24, 52) == 0);
}

TEST_CASE("Bananza captured BRA.U preserves its predicate and relative target",
          "[video_core][ampere]") {
    // 040074A01BF12000 fragment abb4ff47..., code offset 0x5b0.
    constexpr Instruction Nop{{0x00007918, 0, 0, 0x000fc000}};
    std::array<Instruction, 133> code;
    code.fill(Nop);
    code[91] = Instruction{{0x00008947, 0x00000281, 0x03800000, 0x000fea00}};
    code.back() = EXIT;
    const auto lowered = LowerToMaxwell(code);
    INFO(lowered.error);
    REQUIRE(lowered);
    REQUIRE(lowered.instructions.size() == code.size());
    CHECK(lowered.instructions[91] == 0xe24000001b08000fULL);

    const auto backward = LowerToMaxwell(std::array{
        Nop, Instruction{{0x00008947, 0xffffffe1, 0x0383ffff, 0x000fea00}}, EXIT});
    REQUIRE(backward);
    CHECK(backward.instructions[1] == 0xe2400fffff08000fULL);

    const auto guarded = LowerToMaxwell(std::array{
        Instruction{{0x00008947, 0x11, 0x04800000, 0x000fea00}}, Nop, EXIT});
    REQUIRE(guarded);
    REQUIRE(guarded.instructions.size() == 4);
    CHECK(guarded.instructions[0] == 0xe24000000100000fULL);
    CHECK(guarded.instructions[1] == 0xe24000000109000fULL);
}

TEST_CASE("Ampere BRA.U rejects unsupported fields and invalid targets", "[video_core][ampere]") {
    const Instruction branch{{0x00008947, 0x1, 0x03800000, 0x000fea00}};
    for (const u32 bit : {82U, 83U, 84U, 85U, 86U, 91U, 92U, 93U, 94U, 95U,
                          96U, 97U, 98U, 99U, 100U, 101U, 102U, 103U, 104U}) {
        auto unsupported = branch;
        unsupported.words[bit / 32] |= 1U << (bit % 32);
        const auto lowered = LowerToMaxwell(std::array{unsupported, EXIT});
        INFO(bit);
        REQUIRE_FALSE(lowered);
        CHECK(lowered.error == "native branch convergence mode");
    }
    for (const u32 offset : {5U, 0x11U, 0xffffffe1U}) {
        auto invalid = branch;
        invalid.words[1] = offset;
        if (offset == 0xffffffe1U) invalid.words[2] |= 0x3ffff;
        REQUIRE_FALSE(LowerToMaxwell(std::array{invalid, EXIT}));
    }
}

TEST_CASE("Ampere TLD.SCR.LZ takes its handle from a uniform register", "[video_core][ampere]") {
    // TLD.SCR.LZ R5, R8, R4, R5, UR4, 0x0, 2D, 0x7 -> Maxwell TLD.B.
    const std::array<Instruction, 2> code{{
        {{0x04087f66, 0x30000405, 0x089e0705, 0x001f4200}},
        EXIT,
    }};
    const auto lowered = LowerToMaxwell(code);
    REQUIRE(static_cast<bool>(lowered));
    const u64 tld = FindMaxwell(lowered, 0xdd, 56);
    REQUIRE(tld != 0);
    REQUIRE(((tld >> 28) & 7) == 2); // 2D
    REQUIRE(((tld >> 31) & 0xf) == 7);
}

TEST_CASE("Ampere I2F converts signed 16-bit sources", "[video_core][ampere]") {
    // I2F.S16 R8, R8
    const std::array<Instruction, 2> code{{
        {{0x00087306, 0x00000008, 0x00101400, 0x000fe200}},
        EXIT,
    }};
    const auto lowered = LowerToMaxwell(code);
    REQUIRE(static_cast<bool>(lowered));
    const u64 i2f = FindMaxwell(lowered, 0x5cb, 52);
    REQUIRE(((i2f >> 10) & 3) == 1); // 16-bit source
    REQUIRE(((i2f >> 13) & 1) == 1); // signed
}

TEST_CASE("Ampere integer-to-float conversions take immediates", "[video_core][ampere]") {
    // I2F.U32.RP R7, 0x20; I2FP.F32.U32.RZ R4, 0x100; WARPSYNC R7
    const std::array<Instruction, 4> code{{
        {{0x00077906, 0x00000020, 0x00209000, 0x000e2200}},
        {{0x00047845, 0x00000100, 0x0020d000, 0x000fe200}},
        {{0x00007348, 0x00000007, 0x03800000, 0x000fe800}},
        EXIT,
    }};
    REQUIRE(static_cast<bool>(LowerToMaxwell(code)));
}

TEST_CASE("Ampere SUST.P keeps its channel mask", "[video_core][ampere]") {
    // SUST.P.2D.STRONG.SM.R.IGN [R6], R4, 0x0, 0x59
    const std::array<Instruction, 2> code{{
        {{0x06007b9b, 0x60005904, 0x0010a100, 0x000fe200}},
        EXIT,
    }};
    const auto lowered = LowerToMaxwell(code);
    REQUIRE(static_cast<bool>(lowered));
    REQUIRE(((FindMaxwell(lowered, 0xeb2, 52) >> 20) & 0xf) == 1);
}

TEST_CASE("Ampere SHFL with a register lane keeps its mode", "[video_core][ampere]") {
    // SHFL.DOWN PT, R7, R0, R5, 0x1f
    const std::array<Instruction, 2> code{{
        {{0x00077589, 0x08001f05, 0x000e0000, 0x000fe200}},
        EXIT,
    }};
    const auto lowered = LowerToMaxwell(code);
    REQUIRE(static_cast<bool>(lowered));
    REQUIRE(((FindMaxwell(lowered, 0xef1, 52) >> 30) & 3) == 2);
}

TEST_CASE("Ampere calls find a return address set ahead of them", "[video_core][ampere]") {
    // MOV R9, 0x30; MOV R4, 0x5; CALL.REL.NOINC; EXIT; RET.REL.NODEC R9
    std::array<Instruction, 5> code{{
        {{0x00097802, 0x30, 0x00000f00, 0}},
        {{0x00047802, 0x5, 0x00000f00, 0}},
        {{0x00007944, 0x10, 0x03c00000, 0}},
        EXIT,
        {{0x09007950, 0xffffffb0, 0x03c3ffff, 0}},
    }};
    REQUIRE(static_cast<bool>(LowerToMaxwell(code)));

    // The link register is written again between its MOV and the call.
    code[1] = {{0x00097802, 0x5, 0x00000f00, 0}};
    const auto lowered = LowerToMaxwell(code);
    REQUIRE(!static_cast<bool>(lowered));
    REQUIRE(lowered.error == "native call linkage mode");
}

TEST_CASE("Ampere programs go on past an EXIT that a branch jumps over", "[video_core][ampere]") {
    // A call, then a branch over an EXIT to more of the program, with another call there.
    const std::array<Instruction, 8> code{{
        {{0x00047802, 0x20, 0x00000f00, 0}},
        {{0x00007944, 0x50, 0x03c00000, 0}},
        {{0x00007947, 0x10, 0x03800000, 0}},
        EXIT,
        {{0x00047802, 0x60, 0x00000f00, 0}},
        {{0x00007944, 0x10, 0x03c00000, 0}},
        EXIT,
        {{0x04007950, 0xffffff80, 0x03c3ffff, 0}},
    }};
    const auto lowered = LowerToMaxwell(code);
    REQUIRE(static_cast<bool>(lowered));
    REQUIRE(lowered.source_size == sizeof(code));
}
} // namespace

TEST_CASE("Ocarina LDG into a uniform mirror register lowers through a shadow bank",
          "[video_core][ampere]") {
    using Shader::Ampere::NativeRegisterCopy;
    // LDG.E.64 R228, desc[UR4] (Ocarina's vertex shaders): R228 is UR12's mirror, so the pair
    // lives in IR bank 1 and the load writes a borrowed quartet below R180 (R176).
    const Instruction load{{0xffe47981, 0x00002004, 0x0c1e8b00, 0x000f6800}};
    const auto lowered = LowerToMaxwell(std::array{load, EXIT});
    INFO(lowered.error);
    REQUIRE(static_cast<bool>(lowered));
    const auto& words = lowered.instructions;
    const auto has = [&](u64 word) { return std::find(words.begin(), words.end(), word) != words.end(); };
    for (u32 i = 0; i < 4; ++i) {
        CHECK(has(NativeRegisterCopy(2, 176 + i, 0, 176 + i)));
        CHECK(has(NativeRegisterCopy(0, 176 + i, 1, 228 + i)));
        CHECK(has(NativeRegisterCopy(1, 228 + i, 0, 176 + i)));
        CHECK(has(NativeRegisterCopy(0, 176 + i, 2, 176 + i)));
    }
    const u64 ldg = FindMaxwell(lowered, 0xeed, 52);
    REQUIRE(ldg != 0);
    CHECK((ldg & 0xff) == 176);
    CHECK(((ldg >> 8) & 0xff) == 220); // UR4

    // Without the uniform bank R246 and up are scratch and not quartet aligned: still rejected.
    const Instruction scratch_mov{{0x00fa7802, 0x1234, 0x00000f00, 0}};
    const auto scratch = LowerToMaxwell(std::array{scratch_mov, EXIT});
    REQUIRE_FALSE(static_cast<bool>(scratch));
    CHECK(scratch.error == "native program uses lowering scratch registers");
    // An instruction whose register operands are not known to the shadowing stays rejected.
    const Instruction texture{{0x01e87b60, 0x30800a04, 0x001e03ff, 0x100f4200}};
    const auto unknown = LowerToMaxwell(std::array{load, texture, EXIT});
    REQUIRE_FALSE(static_cast<bool>(unknown));
    CHECK(unknown.error == "native register shadowing mode");
}

TEST_CASE("Hyrule Warriors TLD.LZ on a uniform mirror register lowers through a shadow bank",
          "[video_core][ampere]") {
    using Shader::Ampere::NativeRegisterCopy;
    // TLD.LZ R232, R232, 1D (Hyrule Warriors compute): R232 is UR16's mirror; the fetch reads and
    // writes a borrowed quartet below R180 (R176).
    const Instruction load{{0xffe47981, 0x00002004, 0x0c1e8b00, 0x000f6800}};
    const Instruction texture{{0xe8e87b66, 0x10000cff, 0x009e01ff, 0x000f4200}};
    const auto lowered = LowerToMaxwell(std::array{load, texture, EXIT});
    INFO(lowered.error);
    REQUIRE(static_cast<bool>(lowered));
    const auto& words = lowered.instructions;
    const auto count = [&](u64 word) { return std::count(words.begin(), words.end(), word); };
    for (u32 i = 0; i < 4; ++i) {
        CHECK(count(NativeRegisterCopy(0, 176 + i, 1, 232 + i)) == 1);
        CHECK(count(NativeRegisterCopy(1, 232 + i, 0, 176 + i)) == 1);
    }
    // The texel fetch (Maxwell TLDS) writes and reads the borrowed register.
    const u64 fetch = FindMaxwell(lowered, 0xda, 56);
    REQUIRE(fetch != 0);
    CHECK((fetch & 0xff) == 176);
    CHECK(((fetch >> 8) & 0xff) == 176);
}

namespace {
class LoweredEnvironment final : public Shader::Environment {
public:
    explicit LoweredEnvironment(std::span<const u64> instructions_) : instructions{instructions_} {}

    u64 ReadInstruction(u32 address) override {
        for (u32 i = 0; i < instructions.size(); ++i) {
            if (address == 0x50 + i * 8 + ((i + 1) / 3) * 8) return instructions[i];
        }
        return 0;
    }
    bool HasNativeInstructions() const override { return true; }
    u32 ReadCbufValue(u32, u32) override { return 0; }
    Shader::TextureType ReadTextureType(u32) override { return Shader::TextureType::Color2D; }
    Shader::TexturePixelFormat ReadTexturePixelFormat(u32) override {
        return Shader::TexturePixelFormat::R32_UINT;
    }
    bool IsTexturePixelFormatInteger(u32) override { return true; }
    u32 TextureBoundBuffer() const override { return 0; }
    u32 ReadViewportTransformState() override { return 1; }
    u32 LocalMemorySize() const override { return 0; }
    u32 SharedMemorySize() const override { return 0; }
    std::array<u32, 3> WorkgroupSize() const override { return {64, 1, 1}; }
    bool HasHLEMacroState() const override { return false; }
    std::optional<Shader::ReplaceConstant> GetReplaceConstBuffer(u32, u32) override {
        return std::nullopt;
    }
    void Dump(u64, u64) override {}

private:
    std::span<const u64> instructions;
};
} // namespace

TEST_CASE("Shadowed registers keep their values across instructions that borrow their quartet",
          "[video_core][ampere]") {
    // R228 (UR12's mirror) and R249 (scratch) are guest registers here; each instruction that
    // touches one borrows R176-R179, which the guest also uses.
    const std::array<Instruction, 7> code{{
        {{0xffec7981, 0x00002004, 0x0c1e8b00, 0x000f2200}}, // LDG.E.64 R236, desc[UR4]
        {{0x00b07802, 0x5678, 0x00000f00, 0}},              // MOV R176, 0x5678
        {{0x00e47802, 0x1234, 0x00000f00, 0}},              // MOV R228, 0x1234
        {{0x1bf97221, 0x00000028, 0x00010000, 0x000fe200}}, // FADD R249, R27, R40
        {{0x02007386, 0x000000e4, 0x0010a900, 0}},          // STG.E [R2.64], R228
        {{0x02007386, 0x000000b0, 0x0010a900, 0}},          // STG.E [R2.64], R176
        EXIT,
    }};
    const auto lowered = LowerToMaxwell(code);
    INFO(lowered.error);
    REQUIRE(static_cast<bool>(lowered));

    LoweredEnvironment env{lowered.instructions};
    Shader::ObjectPool<Shader::IR::Inst> pool;
    Shader::IR::Block block{pool};
    Shader::IR::Program program;
    program.blocks = {&block};
    program.post_order_blocks = {&block};
    const u32 last = static_cast<u32>(lowered.instructions.size() - 1);
    Shader::Maxwell::Translate(env, &block, 0x50, 0x50 + last * 8 + ((last + 1) / 3) * 8);
    Shader::Optimization::SsaRewritePass(program);
    Shader::Optimization::ConstantPropagationPass(env, program);
    std::vector<u32> stored;
    for (const Shader::IR::Inst& inst : block) {
        if (inst.GetOpcode() != Shader::IR::Opcode::WriteGlobal32) continue;
        REQUIRE(inst.Arg(1).IsImmediate());
        stored.push_back(inst.Arg(1).U32());
    }
    CHECK(stored == std::vector<u32>{0x1234, 0x5678});
}
