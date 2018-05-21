/*
 * Copyright 2018 Taichi Nishimura
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef BISCOTTI_DEBUG_PRINT_H_
#define BISCOTTI_DEBUG_PRINT_H_

#include "biscotti/stats.h"

namespace biscotti {

void PrintDebug(ProcessStats* stats, std::string s);

}  // namespace biscotti

#define BISCOTTI_LOG(stats, ...)                                    \
  do {                                                             \
    char debug_string[1024];                                       \
    int res = snprintf(debug_string, sizeof(debug_string),         \
                       __VA_ARGS__);                               \
    assert(res > 0 && "expected successful printing");             \
    (void)res;                                                     \
    debug_string[sizeof(debug_string) - 1] = '\0';                 \
    ::biscotti::PrintDebug(                      \
         stats, std::string(debug_string));        \
  } while (0)
#define BISCOTTI_LOG_QUANT(stats, q)                    \
  for (int y = 0; y < 8; ++y) {                        \
    for (int c = 0; c < 3; ++c) {                      \
      for (int x = 0; x < 8; ++x)                      \
        BISCOTTI_LOG(stats, " %2d", (q)[c][8 * y + x]); \
      BISCOTTI_LOG(stats, "   ");                       \
    }                                                  \
    BISCOTTI_LOG(stats, "\n");                          \
  }

#endif  // BISCOTTI_DEBUG_PRINT_H_