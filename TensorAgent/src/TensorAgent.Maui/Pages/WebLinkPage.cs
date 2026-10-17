// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using TensorAgent.Core.Localization;

namespace TensorAgent.Maui.Pages;

/// <summary>An independent browser for hyperlinks opened from the conversation.</summary>
public sealed class WebLinkPage : ContentPage
{
    private readonly WebView _webView;
    private readonly Label _address;
    private readonly Button _back;
    private readonly Button _forward;
    private readonly ActivityIndicator _loading;
    private readonly Label _status;
    private readonly HorizontalStackLayout _statusRow;
    private readonly Func<WebLinkPage, Task> _close;
    private bool _closing;

    public WebLinkPage(Uri uri, string? displayName, Func<WebLinkPage, Task> close)
    {
        if (!uri.IsAbsoluteUri || uri.Scheme is not ("http" or "https"))
            throw new ArgumentException("A web page requires an absolute HTTP or HTTPS address.", nameof(uri));

        _close = close;
        Title = string.IsNullOrWhiteSpace(displayName) ? Loc.T("app.webLink.title") : displayName;
        BackgroundColor = Theme.Background;
        SafeAreaEdges = Microsoft.Maui.SafeAreaEdges.All;
        Shell.SetNavBarIsVisible(this, false);

        _webView = new WebView { BackgroundColor = Theme.Background };
        _address = new Label
        {
            Text = uri.AbsoluteUri, TextColor = Theme.Text, FontSize = 12,
            LineBreakMode = LineBreakMode.MiddleTruncation, VerticalOptions = LayoutOptions.Center,
        };
        _back = BrowserButton("←", Loc.T("app.webLink.back"));
        _forward = BrowserButton("→", Loc.T("app.webLink.forward"));
        _back.IsEnabled = _forward.IsEnabled = false;
        _back.Clicked += (_, _) => { if (_webView.CanGoBack) _webView.GoBack(); };
        _forward.Clicked += (_, _) => { if (_webView.CanGoForward) _webView.GoForward(); };
        var closeButton = BrowserButton(Loc.T("app.webLink.close"), Loc.T("app.webLink.close"));
        closeButton.Clicked += async (_, _) =>
        {
            if (_closing) return;
            _closing = true;
            try { await _close(this); }
            catch (Exception ex) { Console.WriteLine("TensorAgent: close-link failed: " + ex.Message); }
            finally { _closing = false; }
        };

        var toolbar = new Grid
        {
            Padding = new Thickness(12, 8), ColumnSpacing = 8, BackgroundColor = Theme.Surface,
            ColumnDefinitions = { new(GridLength.Auto), new(GridLength.Auto), new(GridLength.Star), new(GridLength.Auto) },
        };
        toolbar.Add(_back, 0); toolbar.Add(_forward, 1); toolbar.Add(_address, 2); toolbar.Add(closeButton, 3);

        _loading = new ActivityIndicator { Color = Theme.Accent, WidthRequest = 18, HeightRequest = 18 };
        _status = new Label { TextColor = Theme.Muted, FontSize = 12, VerticalOptions = LayoutOptions.Center };
        _statusRow = new HorizontalStackLayout
        {
            Padding = new Thickness(12, 6), Spacing = 8, Children = { _loading, _status },
        };
        var layout = new Grid { RowDefinitions = { new(GridLength.Auto), new(GridLength.Auto), new(GridLength.Star) } };
        layout.Add(toolbar, 0, 0); layout.Add(_statusRow, 0, 1); layout.Add(_webView, 0, 2);
        Content = layout;

        _webView.Navigating += (_, args) =>
        {
            if (!Uri.TryCreate(args.Url, UriKind.Absolute, out Uri? next) || next.Scheme is not ("http" or "https"))
            {
                args.Cancel = true;
                ShowStatus(loading: false, failed: true);
                return;
            }
            _address.Text = next.AbsoluteUri;
            ShowStatus(loading: true);
        };
        _webView.Navigated += (_, args) =>
        {
            _back.IsEnabled = _webView.CanGoBack;
            _forward.IsEnabled = _webView.CanGoForward;
            ShowStatus(loading: false, failed: args.Result != WebNavigationResult.Success);
        };
        ShowStatus(loading: true);
        _webView.Source = new UrlWebViewSource { Url = uri.AbsoluteUri };
    }

    private void ShowStatus(bool loading, bool failed = false)
    {
        _loading.IsRunning = _loading.IsVisible = loading;
        _status.Text = failed ? Loc.T("app.webLink.failed") : loading ? Loc.T("app.webLink.loading") : string.Empty;
        _status.TextColor = failed ? Theme.Danger : Theme.Muted;
        _statusRow.IsVisible = loading || failed;
    }

    private static Button BrowserButton(string text, string description)
    {
        var button = new Button
        {
            Text = text, FontSize = 14, TextColor = Theme.Text, BackgroundColor = Theme.Background,
            Padding = new Thickness(10, 6), MinimumHeightRequest = 40, MinimumWidthRequest = 40,
        };
        SemanticProperties.SetDescription(button, description);
        return button;
    }
}
