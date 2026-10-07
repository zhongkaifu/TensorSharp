// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using TensorSharp.AgentHost.Skills;
using TensorSharp.Server.Hosting;

namespace TensorAgent.Core.Hosting;

/// <summary>Resolves an image download from the page to its existing upload file.</summary>
internal static class ImageDownload
{
    private const string UploadPrefix = "/uploads/";

    /// <summary>
    /// Accept only image files under the app's upload directory, including generated
    /// and edited results. A page request cannot select a remote URL or a file outside
    /// that directory, even through an encoded separator or a symbolic link.
    /// </summary>
    internal static bool TryResolve(string uploadDirectory, string? urlOrPath, string pageUrl, out string? fullPath)
    {
        fullPath = null;
        if (string.IsNullOrEmpty(urlOrPath))
            return false;

        string path = urlOrPath;
        if (!path.StartsWith("/", StringComparison.Ordinal))
        {
            if (!Uri.TryCreate(urlOrPath, UriKind.Absolute, out Uri? uri)
                || !Uri.TryCreate(pageUrl, UriKind.Absolute, out Uri? page)
                || uri.UserInfo.Length != 0
                || (uri.Scheme != Uri.UriSchemeHttp && uri.Scheme != Uri.UriSchemeHttps)
                || Uri.Compare(uri, page, UriComponents.SchemeAndServer, UriFormat.SafeUnescaped,
                    StringComparison.OrdinalIgnoreCase) != 0)
                return false;
            path = uri.AbsolutePath;
        }

        int end = path.IndexOfAny(['?', '#']);
        if (end >= 0)
            path = path[..end];
        if (!path.StartsWith(UploadPrefix, StringComparison.Ordinal))
            return false;

        string name = Uri.UnescapeDataString(path[UploadPrefix.Length..]);
        // Uploads are stored as single filenames. Check both platform separators and
        // Windows alternate streams even when this code runs on an Apple device.
        if (string.IsNullOrWhiteSpace(name) || name is "." or ".."
            || name.IndexOfAny(['/', '\\', ':', '\0']) >= 0
            || UploadContentPolicy.Classify(Path.GetExtension(name)) != "image")
            return false;

        try
        {
            return SkillPathGuard.TryResolveExistingFile(uploadDirectory, name, out fullPath, out _);
        }
        catch (Exception ex) when (ex is ArgumentException or IOException or UnauthorizedAccessException or NotSupportedException)
        {
            fullPath = null;
            return false;
        }
    }
}
