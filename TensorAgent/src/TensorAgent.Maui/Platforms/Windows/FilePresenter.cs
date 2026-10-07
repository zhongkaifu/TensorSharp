// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System.Diagnostics;
using TensorAgent.Core.Localization;
using Windows.Storage;
using Windows.Storage.Pickers;

namespace TensorAgent.Maui.Services;

/// <summary>
/// Shows the user a file the model's own code produced: in the app Windows opens that
/// type with, or, for a type nothing is registered for, selected in File Explorer, where
/// it can be copied or saved elsewhere. The page cannot do either itself; see
/// Services/Apple/FilePresenter.cs for why the tap comes to the app.
/// </summary>
internal static class FilePresenter
{
    /// <summary>Save a copy of a generated image to the location the user chooses.</summary>
    public static async Task<string?> SaveAsync(string fullPath)
    {
        if (string.IsNullOrEmpty(fullPath) || !File.Exists(fullPath))
            return Loc.T("app.openFile.missing");

        try
        {
            var picker = new FileSavePicker
            {
                SuggestedStartLocation = PickerLocationId.Downloads,
                SuggestedFileName = Path.GetFileNameWithoutExtension(fullPath),
            };
            string extension = Path.GetExtension(fullPath);
            picker.FileTypeChoices.Add(extension.TrimStart('.').ToUpperInvariant(), [extension]);

            // A WinRT picker needs the desktop window owner; without it the call
            // fails in an unpackaged MAUI app before showing the destination dialog.
            var window = (Microsoft.UI.Xaml.Window)Application.Current!.Windows[0].Handler!.PlatformView!;
            WinRT.Interop.InitializeWithWindow.Initialize(picker, WinRT.Interop.WindowNative.GetWindowHandle(window));
            StorageFile? destination = await picker.PickSaveFileAsync();
            if (destination is null)
                return null;

            StorageFile source = await StorageFile.GetFileFromPathAsync(fullPath);
            await source.CopyAndReplaceAsync(destination);
            return null;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"TensorAgent: saving {Path.GetFileName(fullPath)} failed: {ex.Message}");
            return ex.Message;
        }
    }

    /// <summary>
    /// Open <paramref name="fullPath"/>, or show it in File Explorer when no app opens it.
    /// Returns the failure to report, or null when something was shown.
    /// </summary>
    public static async Task<string?> PresentAsync(string fullPath, string? displayName)
    {
        if (string.IsNullOrEmpty(fullPath) || !File.Exists(fullPath))
            return Loc.T("app.openFile.missing");

        try
        {
            string title = string.IsNullOrEmpty(displayName) ? Path.GetFileName(fullPath) : displayName;
            if (await Launcher.Default.OpenAsync(new OpenFileRequest(title, new ReadOnlyFile(fullPath))))
                return null;

            using Process? explorer = Process.Start(new ProcessStartInfo("explorer.exe")
            {
                ArgumentList = { "/select," + fullPath },
                UseShellExecute = false,
            });
            return null;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"TensorAgent: presenting {Path.GetFileName(fullPath)} failed: {ex.Message}");
            return ex.Message;
        }
    }
}
