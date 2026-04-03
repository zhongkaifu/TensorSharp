// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
using System;

namespace AdvUtils
{
    public static class Logger
    {
        public enum Level { debug, info, warn, err }

        public static void WriteLine(string msg) => Console.WriteLine(msg);
        public static void WriteLine(Level level, string msg) => Console.WriteLine($"[{level}] {msg}");
        public static void WriteLine(Level level, ConsoleColor color, string msg) => Console.WriteLine($"[{level}] {msg}");
    }
}
