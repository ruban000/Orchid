// SPDX-License-Identifier: GPL-3.0-or-later
#include <array>
#include <unordered_map>
#include <catch2/catch_test_macros.hpp>
#include "shader_recompiler/environment.h"
#include "shader_recompiler/frontend/ampere/lower_to_maxwell.h"
#include "shader_recompiler/frontend/ir/basic_block.h"
#include "shader_recompiler/frontend/maxwell/translate/impl/impl.h"
namespace {
using namespace Shader;
using Ampere::Instruction;
class LeaEnvironment final : public Environment {
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

void RunLeaIR(const IR::Block& block, std::array<u32, 256>& regs, std::array<bool, 7>& preds) {
    std::unordered_map<const IR::Inst*, u32> values;
    auto arg = [&](IR::Value v) -> u32 {
        if (v.IsImmediate())
            return v.Type() == IR::Type::U1 ? u32{v.U1()} : v.U32();
        return values.at(v.Inst());
    };
    for (const IR::Inst& inst : block) {
        using O = IR::Opcode;
        u32 result{};
        switch (inst.GetOpcode()) {
        case O::GetRegister:
            result = regs[static_cast<size_t>(inst.Arg(0).Reg())];
            break;
        case O::SetRegister:
            regs[static_cast<size_t>(inst.Arg(0).Reg())] = arg(inst.Arg(1));
            continue;
        case O::GetPred:
            result = preds[static_cast<size_t>(inst.Arg(0).Pred())];
            break;
        case O::BitwiseAnd32:
            result = arg(inst.Arg(0)) & arg(inst.Arg(1));
            break;
        case O::SelectU32:
            result = arg(inst.Arg(0)) ? arg(inst.Arg(1)) : arg(inst.Arg(2));
            break;
        case O::ShiftRightLogical32:
            REQUIRE(arg(inst.Arg(1)) < 32);
            result = arg(inst.Arg(0)) >> arg(inst.Arg(1));
            break;
        case O::SetPred:
            preds[static_cast<size_t>(inst.Arg(0).Pred())] = arg(inst.Arg(1)) != 0;
            continue;
        case O::IAdd32:
            result = arg(inst.Arg(0)) + arg(inst.Arg(1));
            break;
        case O::ShiftLeftLogical32:
            REQUIRE(arg(inst.Arg(1)) < 32);
            result = arg(inst.Arg(0)) << arg(inst.Arg(1));
            break;
        // The current Maxwell shift translator emits a signed negative-count guard.
        case O::SLessThan:
            result = static_cast<s32>(arg(inst.Arg(0))) < static_cast<s32>(arg(inst.Arg(1)));
            break;
        case O::ULessThan:
            result = arg(inst.Arg(0)) < arg(inst.Arg(1));
            break;
        case O::LogicalNot:
            result = !arg(inst.Arg(0));
            break;
        case O::LogicalAnd:
            result = arg(inst.Arg(0)) && arg(inst.Arg(1));
            break;
        default:
            FAIL("Unexpected IR opcode " << static_cast<int>(inst.GetOpcode()));
        }
        values.emplace(&inst, result);
    }
}
constexpr Instruction Cases[] = {{{0xb8c27c11, 0xb, 0x0f8008ff, 0x000fe200}},
                                 {{0xb6be7c11, 0xb, 0x0f8208ff, 0x000fe200}},
                                 {{0xb4c07c11, 0xb, 0x0f8008ff, 0x000fe400}},
                                 {{0xb2c27c11, 0xb, 0x0f8208ff, 0x001fe400}}};
constexpr Instruction Exit{{0x794d, 0, 0x03800000, 0}};
TEST_CASE("Yoshi LEA low-add carry reaches Maxwell IR", "[video_core][yoshi_lea]") {
    LeaEnvironment env;
    for (const auto original : Cases)
        for (bool alias : {false, true}) {
            auto native = original;
            const u32 aidx = (native.words[0] >> 24) & 255, pout = (native.words[2] >> 17) & 7;
            const u32 dest = alias ? aidx : (native.words[0] >> 16) & 255;
            native.words[0] = (native.words[0] & ~0xff0000U) | (dest << 16);
            const auto lowered = Ampere::LowerToMaxwell(std::array{native, Exit});
            INFO(lowered.error);
            REQUIRE(lowered);
            for (u32 a : {0U, 1U, 0x7fffffffU, 0x80000000U, 0xffffffffU})
                for (u32 b : {0U, 1U, 0x7fffffffU, 0x80000000U, 0xffffffffU})
                    for (bool oldp : {false, true}) {
                        std::array<u32, 256> regs{};
                        regs[aidx] = a;
                        regs[227] = b;
                        if (dest != aidx)
                            regs[dest] = 0x55555555;
                        std::array<bool, 7> preds{true, false, true, false, true, false, true};
                        preds[pout] = oldp;
                        const auto oldpreds = preds;
                        regs[239] = 0x5a;
                        for (u64 insn : lowered.instructions) {
                            const u64 op = insn >> 48;
                            if (op == 0xe300)
                                break;
                            REQUIRE(((insn >> 16) & 15) == 7);
                            ObjectPool<IR::Inst> pool;
                            IR::Block block{pool};
                            Maxwell::TranslatorVisitor v{env, block};
                            if (op == 0x5c18)
                                v.ISCADD_reg(insn);
                            else if (op == 0x5b62)
                                v.ISETP_reg(insn);
                            else if (op == 0x5c98)
                                v.MOV_reg(insn);
                            else
                                FAIL("unexpected emitted word");
                            RunLeaIR(block, regs, preds);
                        }
                        const u32 low = static_cast<u32>((u64{a} << 1) + b);
                        const bool carry = (u64{static_cast<u32>(a << 1)} + b) > 0xffffffffULL;
                        CHECK(regs[dest] == low);
                        CHECK(preds[pout] == carry);
                        for (u32 p = 0; p < 7; ++p)
                            if (p != pout)
                                CHECK(preds[p] == oldpreds[p]);
                        CHECK(regs[227] == b);
                        CHECK(regs[239] == 0x5a);
                        // Independent full-64-bit pair oracle; this does not execute the HI
                        // consumer.
                        for (u32 ah : {0U, 1U, 0xffffffffU})
                            for (u32 bh : {0U, 1U, 0xffffffffU}) {
                                const u64 expected =
                                    (((u64{ah} << 32) | a) << 1) + ((u64{bh} << 32) | b);
                                const u32 high =
                                    static_cast<u32>((u64{ah} << 1) + (a >> 31) + bh + u32{carry});
                                CHECK(((u64{high} << 32) | low) == expected);
                            }
                    }
        }
}

// The second instruction in each pair is the actual captured HI.X consumer. Both sequences are
// translated to real Maxwell IR; no GPU execution or full-shader compilation is implied.
TEST_CASE("Yoshi LEA low/high pair preserves 64-bit address and predicates",
          "[video_core][yoshi_lea]") {
    LeaEnvironment env;
    constexpr Instruction highs[] = {{{0xb8c37c11, 10, 0x080f0cb7, 0x000fe400}},
                                     {{0xb6bf7c11, 10, 0x088f0cb5, 0x000fe400}},
                                     {{0xb4c17c11, 10, 0x080f0cb3, 0x000fe400}}};
    for (u32 i = 0; i < 3; ++i) {
        const auto low = Ampere::LowerToMaxwell(std::array{Cases[i], Exit});
        const auto high = Ampere::LowerToMaxwell(std::array{highs[i], Exit});
        INFO(low.error);
        INFO(high.error);
        REQUIRE(low);
        REQUIRE(high);
        const u32 lowreg = (Cases[i].words[0] >> 16) & 255,
                  highreg = (highs[i].words[0] >> 16) & 255;
        const u32 areg = (Cases[i].words[0] >> 24) & 255, creg = highs[i].words[2] & 255,
                  pout = (Cases[i].words[2] >> 17) & 7;
        for (u32 a : {0U, 1U, 0x7fffffffU, 0x80000000U, 0xffffffffU})
            for (u32 b : {0U, 1U, 0x7fffffffU, 0x80000000U, 0xffffffffU})
                for (u32 ah : {0U, 1U, 0xffffffffU})
                    for (u32 bh : {0U, 1U, 0xffffffffU}) {
                        std::array<u32, 256> regs{};
                        regs[areg] = a;
                        regs[creg] = ah;
                        regs[227] = b;
                        regs[226] = bh;
                        regs[239] = 0x5a;
                        std::array<bool, 7> preds{true, false, true, false, true, false, true};
                        const auto saved = preds;
                        for (const auto* program : {&low, &high}) {
                            for (u64 insn : program->instructions) {
                                const u64 op = insn >> 48;
                                if (op == 0xe300)
                                    break;
                                REQUIRE(((insn >> 16) & 15) == 7);
                                ObjectPool<IR::Inst> pool;
                                IR::Block block{pool};
                                Maxwell::TranslatorVisitor v{env, block};
                                if (op == 0x5c18)
                                    v.ISCADD_reg(insn);
                                else if (op == 0x5b62)
                                    v.ISETP_reg(insn);
                                else if (op == 0x5c98)
                                    v.MOV_reg(insn);
                                else if (op == 0x3828)
                                    v.SHR_imm(insn);
                                else if (op == 0x3848)
                                    v.SHL_imm(insn);
                                else if (op == 0x5c10)
                                    v.IADD_reg(insn);
                                else if (op == 0x0100)
                                    v.MOV32I(insn);
                                else if (op == 0x5ca0)
                                    v.SEL_reg(insn);
                                else
                                    FAIL("Unexpected pair instruction " << std::hex << insn);
                                RunLeaIR(block, regs, preds);
                                if (program == &high) {
                                    CHECK(preds[pout] ==
                                          ((u64{static_cast<u32>(a << 1)} + b) > 0xffffffffULL));
                                    for (u32 p = 0; p < 7; ++p)
                                        if (p != pout)
                                            CHECK(preds[p] == saved[p]);
                                }
                            }
                            if (program == &low) {
                                const bool lowcarry =
                                    (u64{static_cast<u32>(a << 1)} + b) > 0xffffffffULL;
                                CHECK(preds[pout] == lowcarry);
                                if (a == 0x80000000U && b == 0)
                                    CHECK_FALSE(preds[pout]);
                            }
                        }
                        const u64 expected = (((u64{ah} << 32) | a) << 1) + ((u64{bh} << 32) | b);
                        CHECK(((u64{regs[highreg]} << 32) | regs[lowreg]) == expected);
                        for (u32 p = 0; p < 7; ++p)
                            if (p != pout)
                                CHECK(preds[p] == saved[p]);
                        CHECK(preds[pout] == ((u64{static_cast<u32>(a << 1)} + b) > 0xffffffffULL));
                        CHECK(regs[226] == bh);
                        CHECK(regs[227] == b);
                        CHECK(regs[239] == 0x5a);
                    }
    }
}
TEST_CASE("Yoshi LEA high input aliases use original instruction values",
          "[video_core][yoshi_lea]") {
    LeaEnvironment env;
    constexpr Instruction original{{0xb8c37c11, 10, 0x080f0cb7, 0x000fe400}};
    for (u32 dest : {195U, 184U, 183U}) {
        auto native = original;
        native.words[0] = (native.words[0] & ~0xff0000U) | (dest << 16);
        const auto lowered = Ampere::LowerToMaxwell(std::array{native, Exit});
        INFO(lowered.error);
        REQUIRE(lowered);
        for (u32 a : {0U, 0x80000000U, 0xffffffffU})
            for (u32 c : {0U, 1U, 0xffffffffU})
                for (bool carry : {false, true}) {
                    std::array<u32, 256> regs{};
                    regs[184] = a;
                    regs[183] = c;
                    regs[226] = 0xffffffffU;
                    regs[239] = 0x5a;
                    std::array<bool, 7> preds{carry, true, true, false, true, false, true};
                    const auto saved = preds;
                    for (u64 insn : lowered.instructions) {
                        const u64 op = insn >> 48;
                        if (op == 0xe300)
                            break;
                        ObjectPool<IR::Inst> pool;
                        IR::Block block{pool};
                        Maxwell::TranslatorVisitor v{env, block};
                        if (op == 0x3828)
                            v.SHR_imm(insn);
                        else if (op == 0x3848)
                            v.SHL_imm(insn);
                        else if (op == 0x5c10)
                            v.IADD_reg(insn);
                        else if (op == 0x0100)
                            v.MOV32I(insn);
                        else if (op == 0x5ca0)
                            v.SEL_reg(insn);
                        else
                            FAIL("unexpected high instruction");
                        RunLeaIR(block, regs, preds);
                    }
                    CHECK(regs[dest] ==
                          static_cast<u32>((u64{c} << 1) + (a >> 31) + 0xffffffffU + u32{carry}));
                    CHECK(preds == saved);
                    CHECK(regs[226] == 0xffffffffU);
                    CHECK(regs[239] == 0x5a);
                }
    }
}
TEST_CASE("Yoshi LEA high keeps unverified encodings rejected", "[video_core][yoshi_lea]") {
    const Instruction original{{0xb8c37c11, 10, 0x080f0cb7, 0x000fe400}};
    for (const auto [word, bit] : std::array{
             std::pair{0U, 12U}, std::pair{1U, 0U}, std::pair{2U, 9U}, std::pair{2U, 10U},
             std::pair{2U, 11U}, std::pair{2U, 12U}, std::pair{2U, 26U}, std::pair{3U, 0U}}) {
        auto x = original;
        x.words[word] ^= 1U << bit;
        CHECK_FALSE(Ampere::LowerToMaxwell(std::array{x, Exit}));
    }
    auto p2 = original;
    p2.words[2] = (p2.words[2] & ~(7U << 23)) | (2U << 23);
    CHECK_FALSE(Ampere::LowerToMaxwell(std::array{p2, Exit}));
    auto rz = original;
    rz.words[2] = (rz.words[2] & ~255U) | 255U;
    CHECK(Ampere::LowerToMaxwell(std::array{rz, Exit}));
}
TEST_CASE("Yoshi LEA retains unverified-form guards", "[video_core][yoshi_lea]") {
    for (const auto original : Cases) {
        for (const auto [word, bit] : std::array{
                 std::pair{0U, 12U}, std::pair{1U, 0U}, std::pair{2U, 0U}, std::pair{2U, 8U},
                 std::pair{2U, 10U}, std::pair{2U, 11U}, std::pair{2U, 12U}, std::pair{3U, 0U}}) {
            auto x = original;
            x.words[word] ^= 1U << bit;
            CHECK_FALSE(Ampere::LowerToMaxwell(std::array{x, Exit}));
        }
        auto hi = original;
        hi.words[2] ^= 1U << 16;
        CHECK(Ampere::LowerToMaxwell(std::array{hi, Exit}));
        auto x = original;
        x.words[2] = (x.words[2] & ~(7U << 17)) | (2U << 17);
        CHECK_FALSE(Ampere::LowerToMaxwell(std::array{x, Exit}));
    }
}
} // namespace
