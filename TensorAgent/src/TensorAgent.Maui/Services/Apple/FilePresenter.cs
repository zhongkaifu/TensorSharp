// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using Foundation;
using QuickLook;
using TensorAgent.Core.Localization;
using UIKit;

namespace TensorAgent.Maui.Services;

/// <summary>
/// Shows the user a file the model's own code produced.
///
/// <para>
/// "Open it or save it" is the whole requirement, and on iOS those are two different
/// system components. QuickLook is the OPEN half — a full-screen viewer that renders
/// PDF, images, plain text, CSV and the Office formats, with the system share button in
/// its own navigation bar, so saving is one tap further on. The share sheet is the SAVE
/// half and the fallback for anything QuickLook will not render (an archive, a database,
/// a file with no useful type), where a viewer would show a blank page and a share sheet
/// still offers Save to Files, Mail and AirDrop.
/// </para>
/// <para>
/// Neither can be reached from the WebView. The artifact route serves its files as
/// attachments on purpose — they were written by a program a model wrote and must never
/// render in the origin that holds the launch token — and a WKWebView with no download
/// delegate does nothing at all with an attachment. So the page hands the tap to the app
/// and the app presents one of these.
/// </para>
/// </summary>
internal static class FilePresenter
{
    /// <summary>Export a copy of an image to Files on iOS or a chosen folder on Mac.</summary>
    public static async Task<string?> SaveAsync(string fullPath)
    {
        if (string.IsNullOrEmpty(fullPath) || !File.Exists(fullPath))
            return Loc.T("app.openFile.missing");

        try
        {
            UIViewController host = TopViewController()
                ?? throw new InvalidOperationException("The save dialog could not find an active window.");
            // Exporting as a copy preserves the cached result that saved chats use.
            // A share sheet on Mac Catalyst does not necessarily offer Save to Files.
            var picker = new UIDocumentPickerViewController([NSUrl.FromFilename(fullPath)], true);
            await host.PresentViewControllerAsync(picker, true);
            return null;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"TensorAgent: saving {Path.GetFileName(fullPath)} failed: {ex.Message}");
            return ex.Message;
        }
    }

    /// <summary>
    /// Preview <paramref name="fullPath"/>, or share it when it cannot be previewed.
    /// Returns the failure to report, or null when something was presented.
    /// </summary>
    public static async Task<string?> PresentAsync(string fullPath, string? displayName)
    {
        if (string.IsNullOrEmpty(fullPath) || !File.Exists(fullPath))
            return Loc.T("app.openFile.missing");

        try
        {
            if (TryPreview(fullPath))
                return null;

            await Share.Default.RequestAsync(new ShareFileRequest
            {
                Title = string.IsNullOrEmpty(displayName) ? Path.GetFileName(fullPath) : displayName,
                File = new ShareFile(fullPath),
            });
            return null;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"TensorAgent: presenting {Path.GetFileName(fullPath)} failed: {ex.Message}");
            return ex.Message;
        }
    }

    /// <summary>
    /// The retained data source. QLPreviewController holds its data source WEAKLY, so a
    /// source that is only a local goes away between the present call and the first
    /// callback and the viewer opens on nothing.
    /// </summary>
    private static SingleFileSource? _source;

    private static bool TryPreview(string fullPath)
    {
        NSUrl url = NSUrl.FromFilename(fullPath);
        if (!QLPreviewController.CanPreviewItem(url))
            return false;

        UIViewController? host = TopViewController();
        if (host is null)
            return false;

        _source = new SingleFileSource(url);
        var preview = new QLPreviewController { DataSource = _source };
        host.PresentViewController(preview, animated: true, completionHandler: null);
        return true;
    }

    /// <summary>
    /// The controller a modal has to be presented from: the key window's root, walked
    /// down through anything already presented. Presenting from a controller that is
    /// itself covered is the classic "Attempt to present … whose view is not in the
    /// window hierarchy" warning, after which nothing appears.
    /// </summary>
    private static UIViewController? TopViewController()
    {
        UIWindow? window = UIApplication.SharedApplication.ConnectedScenes
            .OfType<UIWindowScene>()
            .SelectMany(scene => scene.Windows)
            .FirstOrDefault(w => w.IsKeyWindow)
            ?? UIApplication.SharedApplication.ConnectedScenes
                .OfType<UIWindowScene>()
                .SelectMany(scene => scene.Windows)
                .FirstOrDefault();

        UIViewController? controller = window?.RootViewController;
        while (controller?.PresentedViewController is { } presented)
            controller = presented;
        return controller;
    }

    private sealed class SingleFileSource(NSUrl url) : QLPreviewControllerDataSource
    {
        private readonly QLPreviewItemUrl _item = new(url, url.LastPathComponent ?? string.Empty);

        public override nint PreviewItemCount(QLPreviewController controller) => 1;

        public override IQLPreviewItem GetPreviewItem(QLPreviewController controller, nint index) => _item;
    }

    /// <summary>One file, named the way the user knows it.</summary>
    private sealed class QLPreviewItemUrl(NSUrl url, string title) : QLPreviewItem
    {
        public override NSUrl PreviewItemUrl { get; } = url;

        public override string PreviewItemTitle { get; } = title;
    }
}
