// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;

namespace TensorSharp.Models
{
    /// <summary>Recognizes DFlash/DFlash2/DSpine drafter files among <c>--draft-model</c> paths.</summary>
    internal static class DFlashDraftFiles
    {
        /// <summary>
        /// <paramref name="path"/> when it names a DFlash drafter GGUF (general.architecture = dflash), else null: the same
        /// flag also names an MTP-only file for other architectures, so the architecture string decides, not the extension.
        /// </summary>
        internal static string Resolve(string path)
        {
            if (string.IsNullOrWhiteSpace(path) || !System.IO.File.Exists(path))
                return null;
            try
            {
                using var probe = new GgufFile(path);
                string arch = probe.GetString("general.architecture") ?? string.Empty;
                return string.Equals(arch, DFlashConfig.ArchName, StringComparison.Ordinal) ? path : null;
            }
            catch
            {
                // A file we cannot open is not a drafter; the loader says so if the operator really meant it.
                return null;
            }
        }
    }
}
