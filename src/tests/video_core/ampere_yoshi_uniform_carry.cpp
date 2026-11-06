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
class ProbeEnvironment final : public Environment {
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

void Run(const IR::Block& block, std::array<u32, 256>& regs, std::array<bool, 7>& preds) {
    std::unordered_map<const IR::Inst*, u32> vals;
    auto arg = [&](IR::Value x) -> u32 {
        return x.IsImmediate() ? (x.Type() == IR::Type::U1 ? u32{x.U1()} : x.U32())
                               : vals.at(x.Inst());
    };
    for (const IR::Inst& i : block) {
        using O = IR::Opcode;
        u32 v{};
        switch (i.GetOpcode()) {
        case O::GetRegister:
            v = regs[static_cast<size_t>(i.Arg(0).Reg())];
            break;
        case O::SetRegister:
            regs[static_cast<size_t>(i.Arg(0).Reg())] = arg(i.Arg(1));
            continue;
        case O::GetPred:
            v = preds[static_cast<size_t>(i.Arg(0).Pred())];
            break;
        case O::SetPred:
            preds[static_cast<size_t>(i.Arg(0).Pred())] = arg(i.Arg(1)) != 0;
            continue;
        case O::IAdd32:
            v = arg(i.Arg(0)) + arg(i.Arg(1));
            break;
        case O::IEqual:
            v = arg(i.Arg(0)) == arg(i.Arg(1));
            break;
        case O::UGreaterThanEqual:
            v = arg(i.Arg(0)) >= arg(i.Arg(1));
            break;
        case O::ShiftRightLogical32:
            v = arg(i.Arg(1)) < 32 ? arg(i.Arg(0)) >> arg(i.Arg(1)) : 0;
            break;
        case O::ULessThan:
            v = arg(i.Arg(0)) < arg(i.Arg(1));
            break;
        case O::BitwiseOr32:
            v = arg(i.Arg(0)) | arg(i.Arg(1));
            break;
        case O::BitwiseAnd32:
            v = arg(i.Arg(0)) & arg(i.Arg(1));
            break;
        case O::BitFieldUExtract: {
            u32 width = arg(i.Arg(2));
            v = (arg(i.Arg(0)) >> arg(i.Arg(1))) & (width == 32 ? ~0U : ((1U << width) - 1));
            break;
        }
        case O::GetZeroFromOp:
            v = arg(i.Arg(0)) == 0;
            break;
        case O::SelectU1:
        case O::SelectU32:
            v = arg(i.Arg(0)) ? arg(i.Arg(1)) : arg(i.Arg(2));
            break;
        case O::LogicalAnd:
            v = arg(i.Arg(0)) && arg(i.Arg(1));
            break;
        case O::LogicalNot:
            v = !arg(i.Arg(0));
            break;
        default:
            FAIL("unexpected IR opcode " << IR::NameOf(i.GetOpcode()));
        }
        vals.emplace(&i, v);
    }
}

void Execute(const std::vector<u64>& words, std::array<u32, 256>& regs,
             std::array<bool, 7>& preds) {
    ProbeEnvironment env;
    for (u64 w : words) {
        auto op = w >> 48;
        if (op == 0xe300)
            break;
        REQUIRE(((w >> 16) & 15) == 7); // The modeled execution is PT-only.
        ObjectPool<IR::Inst> pool;
        IR::Block block{pool};
        Maxwell::TranslatorVisitor v{env, block};
        if (op == 0x5c10)
            v.IADD_reg(w);
        else if (op == 0x5b62)
            v.ISETP_reg(w);
        else if (op == 0x5c98)
            v.MOV_reg(w);
        else if (op == 0x38e8)
            v.P2R_imm(w);
        else if (op == 0x38f0)
            v.R2P_imm(w);
        else if (op == 0x3800)
            v.BFE_imm(w);
        else if (op == 0x5ca0)
            v.SEL_reg(w);
        else
            FAIL("unexpected lowered word " << std::hex << w);
        Run(block, regs, preds);
    }
}
TEST_CASE("captured uniform carry pair preserves both predicate banks",
          "[yoshi_uniform_carry_ir]") {
    constexpr Instruction producer{{0x0b0b7290, 4, 0x0f83e00d, 0xfe200}},
        consumer{{0x0a0a7290, 5, 0x08fe0410, 0xfe400}}, end{{0x794d, 0, 0x03800000, 0}};
    const auto low = Ampere::LowerToMaxwell(std::array{producer, end});
    INFO(low.error);
    REQUIRE(low);
    const auto high = Ampere::LowerToMaxwell(std::array{consumer, end});
    INFO(high.error);
    REQUIRE(high);
    const std::array<std::array<u32, 3>, 4> inputs{{{{0, 0, 0}},
                                                    {{0xffffffffU, 1, 0}},
                                                    {{0, 1, 0xffffffffU}},
                                                    {{0xffffffffU, 0xffffffffU, 2}}}};
    for (auto in : inputs)
        for (u32 initial_up : {0xa5a55a95U, 0xa5a55ab5U, 0xa5a55ad5U, 0xa5a55af5U})
            for (unsigned mask = 0; mask < 128; ++mask) {
                std::array<u32, 256> r{};
                r[227] = in[0];
                r[220] = in[1];
                r[229] = in[2];
                r[226] = 0xffffffffU;
                r[221] = 0x11223344;
                r[232] = 0x55667788;
                r[239] = initial_up;
                std::array<bool, 7> p{};
                for (unsigned i = 0; i < 7; ++i)
                    p[i] = (mask >> i) & 1;
                auto old = p;
                u32 old_up = r[239];
                const u64 ab = u64{in[0]} + in[1];
                const u64 abc = u64{u32(ab)} + in[2];
                u32 c1 = ab >> 32, c2 = abc >> 32;
                Execute(low.instructions, r, p);
                CHECK(r[227] == u32(abc));
                CHECK(r[239] == ((old_up & ~0x60U) | (c1 << 5) | (c2 << 6)));
                CHECK(p == old);
                // Exercise an ordinary P1-guarded selection between the two uniform ops.
                r[10] = 0x1234;
                r[11] = 0x5678;
                const u64 select =
                    0x5ca0000000070000ULL | 12 | (u64{10} << 8) | (u64{11} << 20) | (u64{1} << 39);
                Execute({select}, r, p);
                CHECK(r[12] == (((mask >> 1) & 1) ? 0x1234U : 0x5678U));
                CHECK(p == old);
                Execute(high.instructions, r, p);
                CHECK(r[226] == u32(u64{0xffffffffU} + 0x11223344U + 0x55667788U + c1 + c2));
                const u64 expected = ((u64{0xffffffffU} << 32) | in[0]) +
                                     ((u64{0x11223344U} << 32) | in[1]) +
                                     ((u64{0x55667788U} << 32) | in[2]);
                CHECK(((u64{r[226]} << 32) | r[227]) == expected);
                CHECK(p == old);
                CHECK(r[239] == ((old_up & ~0x60U) | (c1 << 5) | (c2 << 6)));
            }
}
TEST_CASE("uniform carry inputs and destination aliases", "[yoshi_uniform_carry_ir]") {
    constexpr Instruction end{{0x794d, 0, 0x03800000, 0}};
    for (u32 dst : {11U, 4U, 13U}) {
        Instruction word{{0x0b0b7290, 4, 0x0f83e00d, 0xfe200}};
        word.words[0] = (word.words[0] & ~(0xffU << 16)) | (dst << 16);
        const auto out = Ampere::LowerToMaxwell(std::array{word, end});
        REQUIRE(out);
        std::array<u32, 256> r{};
        r[227] = 0xffffffffU;
        r[220] = 0xffffffffU;
        r[229] = 2;
        r[239] = 0xaabbcc15U;
        std::array<bool, 7> p{true, false, true, false, true, false, true};
        const auto old = p;
        Execute(out.instructions, r, p);
        CHECK(r[216 + dst] == 0);
        CHECK(r[239] == 0xaabbcc75U);
        CHECK(p == old);
    }
    for (u32 dst : {10U, 5U, 16U}) {
        Instruction word{{0x0a0a7290, 5, 0x08fe0410, 0xfe400}};
        word.words[0] = (word.words[0] & ~(0xffU << 16)) | (dst << 16);
        const auto out = Ampere::LowerToMaxwell(std::array{word, end});
        REQUIRE(out);
        std::array<u32, 256> r{};
        r[226] = 0xffffffffU;
        r[221] = 0xffffffffU;
        r[232] = 2;
        r[239] = 0xaabbcc75U;
        std::array<bool, 7> p{true, false, true, false, true, false, true};
        const auto old = p;
        Execute(out.instructions, r, p);
        CHECK(r[216 + dst] == 2);
        CHECK(r[239] == 0xaabbcc75U);
        CHECK(p == old);
    }
}
TEST_CASE("uniform carry consumer keeps other modes rejected", "[yoshi_uniform_carry_ir]") {
    constexpr Instruction end{{0x794d, 0, 0x03800000, 0}};
    const Instruction original{{0x0a0a7290, 5, 0x08fe0410, 0xfe400}};
    for (u32 bit : {12U, 72U, 75U, 80U, 81U, 84U, 90U}) {
        Instruction word = original;
        word.words[bit / 32] ^= 1U << (bit % 32);
        CHECK_FALSE(Ampere::LowerToMaxwell(std::array{word, end}));
    }
    Instruction high_register = original;
    high_register.words[1] = 22;
    CHECK_FALSE(Ampere::LowerToMaxwell(std::array{high_register, end}));
}
} // namespace
