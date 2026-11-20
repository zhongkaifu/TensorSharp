// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using TensorAgent.Core.Catalog;
using TensorAgent.Core.Localization;

namespace TensorAgent.Maui.Pages;

/// <summary>
/// The terms of a catalog entry that asks for them before its first download
/// (<see cref="CatalogModel.LicenseTerms"/>): a summary in the interface language, the license text itself,
/// verbatim, and a link to it at its publisher. Shown modally; <see cref="AskAsync"/> answers whether the user
/// accepted. Leaving the page any other way than Accept is a decline: nothing downloads without a yes.
/// </summary>
public sealed class LicensePage : ContentPage
{
    private readonly TaskCompletionSource<bool> _answer = new(TaskCreationOptions.RunContinuationsAsynchronously);

    private LicensePage(CatalogModel model, CatalogLicenseTerms terms, bool changed)
    {
        Title = Loc.T("models.license.title");
        BackgroundColor = Theme.Background;
        SafeAreaEdges = Microsoft.Maui.SafeAreaEdges.All;
        AutomationId = "license-" + model.Id;

        var heading = new Label
        {
            Text = terms.Title, TextColor = Theme.Text, FontSize = 20, FontAttributes = FontAttributes.Bold,
        };
        var intro = new Label
        {
            Text = Loc.T("models.license.intro", ("model", model.DisplayName), ("license", terms.Title)),
            TextColor = Theme.Text, FontSize = 14,
        };
        var content = new VerticalStackLayout { Spacing = 12, Padding = new Thickness(16), Children = { heading, intro } };
        if (changed)
        {
            content.Children.Add(new Label
            {
                Text = Loc.T("models.license.changed"), TextColor = Theme.Danger, FontSize = 14, FontAttributes = FontAttributes.Bold,
            });
        }
        content.Children.Add(Section(Loc.T("models.license.summary"), terms.Summary, 14));
        content.Children.Add(Section(Loc.T("models.license.fullText"), terms.ReadText(), 12));

        var open = new Button
        {
            Text = Loc.T("models.license.open"), BackgroundColor = Colors.Transparent, TextColor = Theme.Accent,
            HorizontalOptions = LayoutOptions.Start, Padding = new Thickness(0),
        };
        open.Clicked += async (_, _) =>
        {
            try { await Launcher.OpenAsync(terms.Url); }
            catch (Exception ex) { Console.WriteLine("TensorAgent: opening the license failed: " + ex.Message); }
        };
        content.Children.Add(open);

        var decline = new Button
        {
            Text = Loc.T("models.license.decline"), BackgroundColor = Theme.Surface, TextColor = Theme.Text,
            AutomationId = "license-decline",
        };
        var accept = new Button
        {
            Text = Loc.T("models.license.accept"), BackgroundColor = Theme.Accent, TextColor = Colors.White,
            AutomationId = "license-accept",
        };
        decline.Clicked += async (_, _) => await AnswerAsync(false);
        accept.Clicked += async (_, _) => await AnswerAsync(true);
        var buttons = new Grid
        {
            Padding = new Thickness(16, 8), ColumnSpacing = 12, BackgroundColor = Theme.Surface,
            ColumnDefinitions = { new(GridLength.Star), new(GridLength.Star) },
        };
        buttons.Add(decline, 0);
        buttons.Add(accept, 1);

        var layout = new Grid { RowDefinitions = { new(GridLength.Star), new(GridLength.Auto) } };
        layout.Add(new ScrollView { Content = content }, 0, 0);
        layout.Add(buttons, 0, 1);
        Content = layout;
    }

    /// <summary>
    /// Show <paramref name="model"/>'s terms over <paramref name="navigation"/> and wait for the answer: true only
    /// for Accept. <paramref name="changed"/> says the user accepted an earlier revision, so the page says the
    /// terms have changed. Main thread only.
    /// </summary>
    public static async Task<bool> AskAsync(INavigation navigation, CatalogModel model, bool changed)
    {
        ArgumentNullException.ThrowIfNull(navigation);
        CatalogLicenseTerms terms = model.LicenseTerms
            ?? throw new InvalidOperationException(Loc.T("host.license.notRequired", ("model", model.DisplayName)));
        var page = new LicensePage(model, terms, changed);
        await navigation.PushModalAsync(page);
        return await page._answer.Task;
    }

    /// <summary>The system's back gesture or button is a decline, not a way around the question.</summary>
    protected override bool OnBackButtonPressed()
    {
        _ = AnswerAsync(false);
        return true;
    }

    protected override void OnDisappearing()
    {
        base.OnDisappearing();
        // Dismissed by anything else (a sheet swiped away): no.
        _answer.TrySetResult(false);
    }

    private async Task AnswerAsync(bool accepted)
    {
        // Answered before the page closes: closing raises OnDisappearing, whose "no" must not win over a yes.
        if (!_answer.TrySetResult(accepted))
            return;
        try { await Navigation.PopModalAsync(); }
        catch (Exception ex) { Console.WriteLine("TensorAgent: closing the license failed: " + ex.Message); }
    }

    private static View Section(string title, string body, double size) => new VerticalStackLayout
    {
        Spacing = 6,
        Children =
        {
            new Label { Text = title, TextColor = Theme.Muted, FontSize = 13, FontAttributes = FontAttributes.Bold },
            new Border
            {
                BackgroundColor = Theme.Surface, StrokeThickness = 0, Padding = new Thickness(12),
                StrokeShape = new Microsoft.Maui.Controls.Shapes.RoundRectangle { CornerRadius = 10 },
                Content = new Label { Text = body, TextColor = Theme.Text, FontSize = size },
            },
        },
    };
}
