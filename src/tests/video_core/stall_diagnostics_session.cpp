// SPDX-FileCopyrightText: Copyright 2026 Orchid Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <cstdlib>
#include <string_view>

#include "video_core/stall_diagnostics.h"

namespace {
using namespace VideoCommon::StallDiagnostics;
Scope* late_scope{};

void ReleaseLateScope() {
    // Registered before Start: Session's static destructor runs FIRST. A scope in a
    // blocked driver call may likewise unwind after sampler shutdown at process exit.
    delete late_scope;
}
} // namespace

int main(int argc, char** argv) {
    const bool expected = argc == 2 && std::string_view{argv[1]} == "enabled";
    std::atexit(ReleaseLateScope);
    Start();
    if (IsActive() != expected) {
        return 1;
    }
    if (expected) {
        late_scope = new Scope{{Phase::GraphicsCreate, "teardown-test", 42}};
    }
    // No logger initialization or emulator/profile creation, and no GPU calls.
    return 0;
}
