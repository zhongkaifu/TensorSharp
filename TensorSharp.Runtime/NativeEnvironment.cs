// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
#nullable enable
using System;
using System.Runtime.InteropServices;

namespace TensorSharp.Runtime
{
    /// <summary>
    /// Sets an environment variable where BOTH managed code and the native library see it.
    /// On Unix, .NET keeps its own copy of the environment: <see cref="Environment.SetEnvironmentVariable(string, string)"/>
    /// does not change the C runtime's table, so a variable the native library reads with
    /// <c>getenv</c> (the ggml ops' tuning switches) never saw a value published that way.
    /// </summary>
    public static class NativeEnvironment
    {
        [DllImport("libc", EntryPoint = "setenv", CharSet = CharSet.Ansi)]
        private static extern int SetEnvUnix(string name, string value, int overwrite);

        [DllImport("libc", EntryPoint = "unsetenv", CharSet = CharSet.Ansi)]
        private static extern int UnsetEnvUnix(string name);

        [DllImport("ucrtbase", EntryPoint = "_putenv_s", CharSet = CharSet.Ansi)]
        private static extern int PutEnvWindows(string name, string value);

        /// <summary>Set (or, with null, remove) <paramref name="name"/> in the managed
        /// environment and in the C runtime's.</summary>
        public static void Set(string name, string? value)
        {
            Environment.SetEnvironmentVariable(name, value);
            try
            {
                if (OperatingSystem.IsWindows())
                    _ = PutEnvWindows(name, value ?? string.Empty);   // "" removes it
                else if (value == null)
                    _ = UnsetEnvUnix(name);
                else
                    _ = SetEnvUnix(name, value, 1);
            }
            catch (DllNotFoundException) { }
            catch (EntryPointNotFoundException) { }
        }
    }
}
