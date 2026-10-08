// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System.Collections.ObjectModel;
using System.Linq;
using TensorAgent.Core.Catalog;
using TensorAgent.Core.Downloads;
using TensorAgent.Core.Hosting;
using TensorAgent.Core.Localization;
using TensorAgent.Core.Settings;
using TensorAgent.Maui.Hosting;

namespace TensorAgent.Maui.Pages;

/// <summary>
/// The built-in model list: what this device can run, what is already on it, and
/// what a download would cost.
///
/// <para>
/// This page has no counterpart in the desktop Web UI, and it cannot have one. The
/// server's model picker lists files an operator put on a disk; here the app is
/// responsible for getting them, which means telling the user the size before they
/// commit to it, resuming an interrupted download rather than starting again, and
/// making a partly-downloaded model obviously partly downloaded.
/// </para>
/// </summary>
public sealed class ModelsPage : ContentPage
{
    private readonly AgentAppHost _app;

    /// <summary>The running app, so the debug reproduction hook can drive the same path a tap does.</summary>
    internal AgentAppHost Host => _app;
    // Keep every row alive, including collapsed/filtered models: a background download
    // must still update its state and complete through the same selection path.
    private readonly ObservableCollection<ModelRow> _rows = new();
    private readonly ObservableCollection<object> _items = new();
    private readonly Dictionary<string, ModelFamilyRow> _families = new(StringComparer.Ordinal);
    private readonly HashSet<string> _expandedFamilies = new(StringComparer.Ordinal);
    private readonly Dictionary<ModelBrowseFilter, Button> _filterButtons = new();
    private ModelBrowseFilter _filter;
    private string _query = string.Empty;
    private bool _initializedExpansion;
    private bool _selecting;
    private string? _loadingModelId;
    private CollectionView _list = null!;
    private SearchBar _search = null!;
    private Label _summary = null!;
    private Button _expandAll = null!;
    private Button _selected = null!;
    private Button _clearSearch = null!;
    private Microsoft.Maui.Dispatching.IDispatcherTimer? _searchTimer;
    private IReadOnlyList<ModelFamilyGroup> _shownGroups = Array.Empty<ModelFamilyGroup>();
    // A normal Download tap means "use this when it finishes" only while the user
    // remains on this page and has not chosen another model in the meantime.
    private string? _pendingAutoSelectId;

    /// <summary>
    /// True only while this page is on screen.
    ///
    /// <para>
    /// A download now outlives the page, so a job that finishes while the user is in
    /// the chat must not drag them back here to load a model they may no longer want.
    /// The automatic "downloaded, so use it" step happens only when they are still
    /// looking at the list they started it from.
    /// </para>
    /// </summary>
    private bool _visible;

    public ModelsPage(LoopbackWebHost host)
    {
        _app = host.App;
        BackgroundColor = Theme.Background;
        Padding = new Thickness(0);
        Build();

        // A new language rebuilds the screen, cells and all. The rows are rebuilt in it by
        // the next Refresh, which appearing always runs.
        Loc.Changed += () => Dispatcher.Dispatch(() =>
        {
            try
            {
                Build();
                if (_visible)
                    Refresh();
            }
            catch (Exception ex)
            {
                Console.WriteLine("TensorAgent: the models list failed to repaint in the new language: " + ex);
            }
        });
    }

    private void Build()
    {
        _searchTimer?.Stop();
        Title = Loc.T("models.title");

        _list = new CollectionView
        {
            AutomationId = "ModelsList",
            ItemsSource = _items,
            ItemTemplate = new BrowserTemplateSelector(new DataTemplate(BuildCell), new DataTemplate(BuildFamilyCell)),
            SelectionMode = SelectionMode.None,
            BackgroundColor = Theme.Background,
            EmptyView = EmptyResults(),
        };

        Content = new Grid
        {
            BackgroundColor = Theme.Background,
            RowDefinitions = { new RowDefinition(GridLength.Auto), new RowDefinition(GridLength.Star) },
            Children =
            {
                Header(),
                _list,
            },
        };
        Grid.SetRow((View)((Grid)Content).Children[1], 1);
    }

    private View Header()
    {
        var header = new VerticalStackLayout { Spacing = 6, Padding = new Thickness(16, 8, 16, 4) };
        header.Children.Add(new Label
        {
            Text = Loc.T("models.header", ("memory", _app.Paths.DeviceMemoryGB)),
            TextColor = Theme.Muted,
            FontSize = 13,
        });
        _search = new SearchBar
        {
            AutomationId = "ModelSearch",
            Placeholder = Loc.T("models.search.placeholder"),
            Text = _query,
            TextColor = Theme.Text,
            PlaceholderColor = Theme.Muted,
            CancelButtonColor = Theme.Accent,
            BackgroundColor = Theme.Surface,
            MinimumHeightRequest = 44,
        };
        SemanticProperties.SetHint(_search, Loc.T("models.search.hint"));
        _searchTimer = Dispatcher.CreateTimer();
        _searchTimer.Interval = TimeSpan.FromMilliseconds(150);
        _searchTimer.IsRepeating = false;
        _searchTimer.Tick += (_, _) => ApplyBrowser(scrollToTop: true);
        _search.TextChanged += (_, e) =>
        {
            _query = e.NewTextValue ?? string.Empty;
            _searchTimer.Stop();
            if (string.IsNullOrWhiteSpace(_query)) ApplyBrowser(scrollToTop: true);
            else _searchTimer.Start();
        };
        // Enter reveals results; only an explicit Download/Use action starts work.
        _search.SearchButtonPressed += (_, _) =>
        {
            ApplyBrowser(scrollToTop: true);
            _search.Unfocus();
        };
        header.Children.Add(_search);

        _filterButtons.Clear();
        var filters = new FlexLayout { Wrap = Microsoft.Maui.Layouts.FlexWrap.Wrap };
        foreach (var option in new[]
        {
            (ModelBrowseFilter.All, "models.filter.all"),
            (ModelBrowseFilter.Compatible, "models.filter.compatible"),
            (ModelBrowseFilter.Downloaded, "models.filter.downloaded"),
        })
        {
            Button button = BrowserButton(Loc.T(option.Item2));
            button.AutomationId = "ModelFilter" + option.Item1;
            button.Margin = new Thickness(0, 0, 6, 0);
            button.Clicked += (_, _) =>
            {
                _filter = option.Item1;
                ApplyBrowser(scrollToTop: true);
            };
            filters.Children.Add(button);
            _filterButtons.Add(option.Item1, button);
        }
        header.Children.Add(filters);

        _selected = BrowserButton(string.Empty);
        _selected.AutomationId = "ShowSelectedModel";
        _selected.HorizontalOptions = LayoutOptions.Start;
        SemanticProperties.SetHint(_selected, Loc.T("models.browser.showSelected"));
        _selected.Clicked += (_, _) => ShowSelected();
        header.Children.Add(_selected);

        _summary = new Label { FontSize = 12, TextColor = Theme.Muted, VerticalOptions = LayoutOptions.Center };
        _expandAll = BrowserButton(Loc.T("models.browser.expandAll"));
        _expandAll.AutomationId = "ModelExpandAll";
        _expandAll.Clicked += (_, _) =>
        {
            bool collapse = _shownGroups.All(g => _expandedFamilies.Contains(g.Id));
            foreach (var group in _shownGroups)
                if (collapse) _expandedFamilies.Remove(group.Id); else _expandedFamilies.Add(group.Id);
            ApplyBrowser();
        };
        _clearSearch = BrowserButton(Loc.T("models.search.clear"));
        _clearSearch.AutomationId = "ClearModelSearch";
        _clearSearch.Clicked += (_, _) => _search.Text = string.Empty;
        var actions = new HorizontalStackLayout { Children = { _expandAll, _clearSearch } };
        var summary = new Grid { ColumnDefinitions = { new ColumnDefinition(GridLength.Star), new ColumnDefinition(GridLength.Auto) } };
        summary.Children.Add(_summary);
        summary.Children.Add(actions);
        Grid.SetColumn(actions, 1);
        header.Children.Add(summary);
        return header;
    }

    private static Button BrowserButton(string text) => new()
    {
        Text = text, FontSize = 12, TextColor = Theme.Muted, BackgroundColor = Theme.Surface,
        CornerRadius = 8, Padding = new Thickness(10, 5), MinimumHeightRequest = 44,
    };

    private View EmptyResults()
    {
        var reset = BrowserButton(Loc.T("models.browser.reset"));
        reset.TextColor = Theme.Accent;
        reset.AutomationId = "ResetModelSearch";
        reset.Clicked += (_, _) =>
        {
            _filter = ModelBrowseFilter.All;
            _search.Text = string.Empty;
            ApplyBrowser(scrollToTop: true);
        };
        return new VerticalStackLayout
        {
            Spacing = 12, Padding = new Thickness(24), HorizontalOptions = LayoutOptions.Center,
            VerticalOptions = LayoutOptions.Center,
            Children =
            {
                new Label { Text = Loc.T("models.browser.empty.title"), TextColor = Theme.Text, FontSize = 18, HorizontalTextAlignment = TextAlignment.Center },
                new Label { Text = Loc.T("models.browser.empty.hint"), TextColor = Theme.Muted, HorizontalTextAlignment = TextAlignment.Center },
                reset,
            },
        };
    }

    protected override void OnAppearing()
    {
        base.OnAppearing();
        // Subscribe before taking the snapshot. Otherwise a transfer can complete
        // after Refresh sees Running and before the handler is attached, leaving the
        // reconstructed row stuck forever in that stale state.
        _app.Downloads.Changed -= OnDownloadChanged;
        _app.Downloads.Changed += OnDownloadChanged;
        try
        {
            Refresh();
            _visible = true;
        }
        catch (Exception ex)
        {
            _app.Downloads.Changed -= OnDownloadChanged;
            // A page that cannot list the models is still a page the user reached. Left
            // to propagate, this cancels the push and drops them back on the chat with
            // nothing said -- indistinguishable from the menu not working.
            Console.WriteLine("TensorAgent: the models list failed to appear: " + ex);
        }
    }

    protected override void OnDisappearing()
    {
        base.OnDisappearing();
        _visible = false;
        _searchTimer?.Stop();
        _pendingAutoSelectId = null;
        _app.Downloads.Changed -= OnDownloadChanged;
    }

    /// <summary>
    /// One report from a running download, from whatever thread the transfer is on.
    ///
    /// <para>
    /// The row is found by id rather than held, because <see cref="Refresh"/> rebuilds
    /// the collection and a captured row would then be updating an object no longer in
    /// the list.
    /// </para>
    /// </summary>
    private void OnDownloadChanged(ModelDownloadStatus status)
    {
        MainThread.BeginInvokeOnMainThread(() =>
        {
            ModelRow? row = _rows.FirstOrDefault(r => string.Equals(r.Model.Id, status.ModelId, StringComparison.Ordinal));
            if (row is null)
                return;

            switch (status.State)
            {
                case DownloadState.Running:
                    if (!row.IsBusy) row.BeginDownload();
                    row.Report(status.Progress);
                    UpdateFamilyActivity(row.Model);
                    return;
                case DownloadState.Completed:
                    bool companionOnly = status.RequestsOnly(CatalogFileRole.Projector)
                        || status.RequestsOnly(CatalogFileRole.Draft);
                    bool selectedNow = string.Equals(
                        _app.Settings.Load().SelectedModelId, row.Model.Id, StringComparison.Ordinal);
                    bool autoSelect = !companionOnly && string.Equals(
                        _pendingAutoSelectId, row.Model.Id, StringComparison.Ordinal);
                    if (autoSelect)
                        _pendingAutoSelectId = null;
                    row.Finish(_app.Models);
                    // A result hidden by a new search/filter or collapsed family must
                    // not interrupt browsing when its background transfer finishes.
                    if (_visible && _items.Contains(row) && (autoSelect || (companionOnly && selectedNow)))
                        Select(row);
                    else if (_visible && companionOnly)
                        Refresh();
                    else
                        ApplyBrowser();
                    return;
                case DownloadState.Cancelled:
                    if (string.Equals(_pendingAutoSelectId, row.Model.Id, StringComparison.Ordinal))
                        _pendingAutoSelectId = null;
                    row.Cancelled(_app.Models);
                    ApplyBrowser();
                    return;
                default:
                    if (string.Equals(_pendingAutoSelectId, row.Model.Id, StringComparison.Ordinal))
                        _pendingAutoSelectId = null;
                    row.Failed(_app.Models, status.Error ?? Loc.T("models.status.downloadFailed"));
                    ApplyBrowser();
                    return;
            }
        });
    }

    /// <summary>
    /// Every built-in entry, runnable ones first.
    ///
    /// <para>
    /// This used to list <c>_app.Catalog</c>, which is <c>ForDevice</c> -- only what
    /// fits. That is the right list for LOADING a model and the wrong one for a page
    /// whose job is to tell the user what exists: on a 12 GB iPhone it silently hid
    /// half the catalog, and when a memory-tier bug made ForDevice return nothing the
    /// page went completely blank with no way to tell "none fit" from "something is
    /// broken". A model that needs a bigger device is shown, greyed, saying so.
    /// </para>
    /// </summary>
    private void Refresh()
    {
        string? selected = _app.Settings.Load().SelectedModelId;
        string? loadedModel = _app.ModelService.LoadedModelName;
        bool visionReady = _app.ModelService.Model?.HasVisionEncoder() ?? false;
        int deviceGB = _app.Paths.DeviceMemoryGB;
        _rows.Clear();
        foreach (CatalogModel model in ModelCatalog.BuiltIn
                     .OrderByDescending(m => m.MinDeviceMemoryGB <= deviceGB)
                     .ThenBy(m => m.MinDeviceMemoryGB)
                     .ThenBy(m => m.TotalBytes))
        {
            _rows.Add(new ModelRow(
                model, _app.Models, selected, deviceGB, _app.Downloads.StatusOf(model.Id),
                loadedModel, visionReady, _app.CatalogDraftHeadAttached));
        }
        if (_loadingModelId is not null)
            _rows.FirstOrDefault(r => r.Model.Id == _loadingModelId)?.BeginLoading();
        foreach (ModelRow row in _rows) row.SelectionInProgress = _selecting;
        if (!_initializedExpansion)
        {
            // Open the current family's branch on first visit; otherwise begin with a
            // compact family overview. Searching never changes these saved choices.
            if (ModelCatalog.Find(selected ?? string.Empty) is { } model)
                _expandedFamilies.Add(ModelBrowser.FamilyId(model));
            _initializedExpansion = true;
        }
        ApplyBrowser();
    }

    private void ApplyBrowser(bool scrollToTop = false)
    {
        if (_summary is null) return;
        _searchTimer?.Stop();
        bool searching = !string.IsNullOrWhiteSpace(_query);
        string? selectedId = _app.Settings.Load().SelectedModelId;
        var installed = _rows.Where(r => r.IsInstalled).Select(r => r.Model.Id).ToHashSet(StringComparer.OrdinalIgnoreCase);
        _shownGroups = ModelBrowser.Browse(_rows.Select(r => r.Model), _query, _filter,
            _app.Paths.DeviceMemoryGB, installed, selectedId);
        var byId = _rows.ToDictionary(r => r.Model.Id, StringComparer.OrdinalIgnoreCase);
        var visible = new List<object>();
        if (searching)
        {
            foreach (CatalogModel model in ModelBrowser.Search(_rows.Select(r => r.Model), _query, _filter,
                         _app.Paths.DeviceMemoryGB, installed, selectedId))
                visible.Add(byId[model.Id]);
        }
        else
        {
            foreach (ModelFamilyGroup group in _shownGroups)
            {
                if (!_families.TryGetValue(group.Id, out ModelFamilyRow? family))
                    _families[group.Id] = family = new ModelFamilyRow(group.Id, group.DisplayName);
                family.Update(group.Models.Select(m => byId[m.Id]).ToArray(), _expandedFamilies.Contains(group.Id));
                visible.Add(family);
                if (family.IsExpanded)
                    visible.AddRange(family.Rows);
            }
        }
        // Retain row/header instances and change only the affected branch. Clearing the
        // collection for a disclosure tap loses scroll position and keyboard focus.
        for (int i = 0; i < visible.Count; i++)
        {
            if (i < _items.Count && ReferenceEquals(_items[i], visible[i])) continue;
            int existing = _items.IndexOf(visible[i]);
            if (existing >= 0) _items.Move(existing, i); else _items.Insert(i, visible[i]);
        }
        while (_items.Count > visible.Count) _items.RemoveAt(_items.Count - 1);

        int count = _shownGroups.Sum(g => g.Models.Count);
        _summary.Text = searching
            ? Loc.Plural("models.browser.results", count)
            : Loc.T("models.browser.summary", ("models", count), ("families", _shownGroups.Count));
        _expandAll.IsVisible = !searching && _shownGroups.Count > 0;
        _expandAll.Text = Loc.T(_shownGroups.All(g => _expandedFamilies.Contains(g.Id))
            ? "models.browser.collapseAll" : "models.browser.expandAll");
        _clearSearch.IsVisible = searching;
        CatalogModel? selectedModel = ModelCatalog.Find(selectedId ?? string.Empty);
        _selected.IsVisible = selectedModel is not null;
        _selected.Text = selectedModel is null ? string.Empty : Loc.T("models.browser.current", ("model", selectedModel.DisplayName));
        foreach (var pair in _filterButtons)
        {
            pair.Value.BackgroundColor = pair.Key == _filter ? Theme.Accent : Theme.Surface;
            pair.Value.TextColor = pair.Key == _filter ? Colors.White : Theme.Muted;
            pair.Value.FontAttributes = pair.Key == _filter ? FontAttributes.Bold : FontAttributes.None;
            SemanticProperties.SetDescription(pair.Value, pair.Value.Text
                + (pair.Key == _filter ? " · " + Loc.T("models.action.selected") : string.Empty));
        }
        if (scrollToTop && _items.Count > 0)
            _list.ScrollTo(0, position: ScrollToPosition.Start, animate: false);
    }

    private void ShowSelected()
    {
        string? id = _app.Settings.Load().SelectedModelId;
        ModelRow? row = _rows.FirstOrDefault(r => r.Model.Id == id);
        if (row is null) return;
        _filter = ModelBrowseFilter.All;
        _expandedFamilies.Add(ModelBrowser.FamilyId(row.Model));
        _search.Text = string.Empty;
        ApplyBrowser();
        _search.Unfocus();
        _list.ScrollTo(row, position: ScrollToPosition.Center, animate: true);
    }

    private void UpdateFamilyActivity(CatalogModel model)
    {
        if (_families.TryGetValue(ModelBrowser.FamilyId(model), out ModelFamilyRow? family))
            family.UpdateActivity();
    }

    private View BuildFamilyCell()
    {
        var disclosure = BrowserButton(string.Empty);
        disclosure.FontSize = 17;
        disclosure.FontAttributes = FontAttributes.Bold;
        disclosure.TextColor = Colors.Transparent;
        disclosure.BackgroundColor = Colors.Transparent;
        disclosure.HorizontalOptions = LayoutOptions.Fill;
        disclosure.VerticalOptions = LayoutOptions.Fill;
        disclosure.SetBinding(Button.TextProperty, nameof(ModelFamilyRow.Title));
        disclosure.SetBinding(AutomationIdProperty, nameof(ModelFamilyRow.AutomationId));
        disclosure.SetBinding(SemanticProperties.DescriptionProperty, nameof(ModelFamilyRow.AccessibleLabel));
        disclosure.SetBinding(SemanticProperties.HintProperty, nameof(ModelFamilyRow.Summary));
        disclosure.Clicked += (_, _) =>
        {
            if (disclosure.BindingContext is not ModelFamilyRow family) return;
            if (!_expandedFamilies.Remove(family.Id)) _expandedFamilies.Add(family.Id);
            ApplyBrowser();
        };
        var title = new Label { FontSize = 17, FontAttributes = FontAttributes.Bold, TextColor = Theme.Text };
        title.SetBinding(Label.TextProperty, nameof(ModelFamilyRow.Title));
        var summary = new Label { FontSize = 12, TextColor = Theme.Muted, Margin = new Thickness(22, 0, 0, 0) };
        summary.SetBinding(Label.TextProperty, nameof(ModelFamilyRow.Summary));
        var activity = new Label { FontSize = 12, TextColor = Theme.Accent, Margin = new Thickness(22, 0, 0, 0) };
        activity.SetBinding(Label.TextProperty, nameof(ModelFamilyRow.Activity));
        activity.SetBinding(IsVisibleProperty, nameof(ModelFamilyRow.IsDownloading));
        var labels = new VerticalStackLayout
        {
            InputTransparent = true, Padding = new Thickness(14, 12), Spacing = 4,
            Children = { title, summary, activity },
        };
        // Native button underneath provides keyboard activation and one accessible
        // expand/collapse action; the labels provide a left-aligned, wrapping layout.
        AutomationProperties.SetExcludedWithChildren(labels, true);
        return new Border
        {
            BackgroundColor = Theme.Surface, StrokeThickness = 0,
            StrokeShape = new Microsoft.Maui.Controls.Shapes.RoundRectangle { CornerRadius = 12 },
            Margin = new Thickness(12, 8, 12, 2),
            Content = new Grid { Children = { disclosure, labels } },
        };
    }

    private View BuildCell()
    {
        var family = new Label { FontSize = 11, TextColor = Theme.Muted };
        family.SetBinding(Label.TextProperty, nameof(ModelRow.FamilyAndMemory));
        var title = new Label { FontSize = 16, TextColor = Theme.Text, FontAttributes = FontAttributes.Bold };
        title.SetBinding(Label.TextProperty, nameof(ModelRow.Title));

        var subtitle = new Label { FontSize = 12, TextColor = Theme.Muted, LineBreakMode = LineBreakMode.WordWrap };
        subtitle.SetBinding(Label.TextProperty, nameof(ModelRow.Subtitle));

        var status = new Label { FontSize = 12, TextColor = Theme.Accent };
        status.SetBinding(Label.TextProperty, nameof(ModelRow.Status));

        var progress = new ProgressBar { ProgressColor = Theme.Accent, HeightRequest = 3 };
        progress.SetBinding(ProgressBar.ProgressProperty, nameof(ModelRow.Fraction));
        progress.SetBinding(IsVisibleProperty, nameof(ModelRow.IsBusy));

        var action = new Button
        {
            FontSize = 14,
            Padding = new Thickness(14, 6),
            BackgroundColor = Theme.Accent,
            TextColor = Colors.White,
            CornerRadius = 8,
            MinimumHeightRequest = 44,
        };
        action.SetBinding(Button.TextProperty, nameof(ModelRow.ActionLabel));
        action.SetBinding(SemanticProperties.DescriptionProperty, nameof(ModelRow.ActionDescription));
        action.SetBinding(IsEnabledProperty, nameof(ModelRow.CanAct));
        action.SetBinding(AutomationIdProperty, nameof(ModelRow.ActionAutomationId));
        action.SetBinding(Button.BackgroundColorProperty, nameof(ModelRow.ActionColor));
        action.Clicked += (s, _) => OnAction(((Button)s!).BindingContext as ModelRow);

        var addVision = new Button
        {
            Text = Loc.T("models.action.addVision"),
            FontSize = 14,
            Padding = new Thickness(14, 6),
            BackgroundColor = Theme.Accent,
            TextColor = Colors.White,
            CornerRadius = 8,
        };
        addVision.SetBinding(IsVisibleProperty, nameof(ModelRow.CanAddVision));
        addVision.Clicked += (s, _) => OnVisionAction(((Button)s!).BindingContext as ModelRow);

        var addDraft = new Button
        {
            FontSize = 14,
            Margin = new Thickness(0, 0, 8, 8),
            Padding = new Thickness(14, 6),
            BackgroundColor = Theme.Accent,
            TextColor = Colors.White,
            CornerRadius = 8,
        };
        addDraft.SetBinding(Button.TextProperty, nameof(ModelRow.AddDraftLabel));
        addDraft.SetBinding(IsVisibleProperty, nameof(ModelRow.CanAddDraft));
        addDraft.Clicked += (s, _) => OnDraftAction(((Button)s!).BindingContext as ModelRow);

        var remove = new Button
        {
            Text = Loc.T("models.action.delete"),
            FontSize = 14,
            Padding = new Thickness(14, 6),
            BackgroundColor = Theme.Surface,
            TextColor = Theme.Muted,
            CornerRadius = 8,
        };
        remove.SetBinding(IsVisibleProperty, nameof(ModelRow.CanDelete));
        remove.Clicked += (s, _) => OnDelete(((Button)s!).BindingContext as ModelRow);

        // Companion choices wrap on a phone and when translated labels are wider.
        foreach (Button button in new[] { action, addVision, remove })
            button.Margin = new Thickness(0, 0, 8, 8);
        var buttons = new FlexLayout { Wrap = Microsoft.Maui.Layouts.FlexWrap.Wrap,
            Children = { action, addVision, addDraft, remove } };

        var card = new Border
        {
            Margin = new Thickness(26, 5, 12, 5),
            Padding = new Thickness(14),
            BackgroundColor = Theme.Surface,
            StrokeThickness = 1,
            StrokeShape = new Microsoft.Maui.Controls.Shapes.RoundRectangle { CornerRadius = 12 },
            Content = new VerticalStackLayout
            {
                Spacing = 6,
                Children = { family, title, subtitle, status, progress, buttons },
            },
        };
        card.SetBinding(Border.StrokeProperty, nameof(ModelRow.CardStroke));
        card.SetBinding(AutomationIdProperty, nameof(ModelRow.AutomationId));
        return card;
    }

    private async void OnAction(ModelRow? row)
    {
        if (row is null || !row.Runnable)
            return;

        if (_app.Downloads.StatusOf(row.Model.Id) is { IsRunning: true })
        {
            _app.Downloads.Cancel(row.Model.Id);
            return;
        }

        // Downloads expose a Stop action above. Loads and local imports do not:
        // accepting a second tap would queue another multi-gigabyte import behind the
        // store lock or race another model load.
        if (row.IsBusy || _selecting)
            return;

        if (row.IsInstalled)
        {
            Select(row);
            return;
        }

        if (row.Model.SideloadOnly)
        {
            await Import(row);
            return;
        }

        AppSettings settings = _app.Settings.Load();
        if (!settings.AllowCellularDownloads && Services.DeviceState.IsOnCellularOnly())
        {
            await DisplayAlert(
                Loc.T("models.alert.cellular.title"),
                Loc.T("models.alert.cellular.model",
                    ("model", row.Model.DisplayName),
                    ("size", (row.Model.TotalBytes / 1e9).ToString("0.0", Loc.Culture)),
                    ("setting", Loc.T("settings.downloads.cellular.title"))),
                Loc.T("common.ok"));
            return;
        }

        Download(row);
    }

    /// <summary>
    /// Copy a publisher-less, hash-pinned card from the Files picker into the model
    /// store. The store stages and verifies the whole file before replacing anything;
    /// selecting the wrong multi-gigabyte GGUF leaves no loadable partial behind.
    /// </summary>
    private async Task Import(ModelRow row)
    {
        try
        {
            FileResult? picked = await FilePicker.Default.PickAsync(new PickOptions
            {
                PickerTitle = Loc.T("models.import.pickerTitle", ("file", row.Model.Weights.FileName)),
            });
            if (picked is null)
                return;

            row.BeginImport();
            await using Stream source = await picked.OpenReadAsync();
            var progress = new Progress<long>(row.ReportImport);
            await _app.Models.ImportAsync(row.Model, source, progress, CancellationToken.None);
            row.Finish(_app.Models);
            Select(row);
        }
        catch (Exception ex)
        {
            row.Failed(_app.Models, ex.Message);
            await DisplayAlert(Loc.T("models.alert.importFailed.title"), ex.Message, Loc.T("common.ok"));
        }
    }

    /// <summary>
    /// Fetch only the optional projector. The ordinary action remains available for
    /// text-only use, so the global optional-download switch still means what it says;
    /// this second button is explicit consent to add the model's image capability.
    /// </summary>
    private async void OnVisionAction(ModelRow? row)
    {
        if (row is null || !row.CanAddVision)
            return;

        AppSettings settings = _app.Settings.Load();
        if (!settings.AllowCellularDownloads && Services.DeviceState.IsOnCellularOnly())
        {
            await DisplayAlert(
                Loc.T("models.alert.cellular.title"),
                Loc.T("models.alert.cellular.vision",
                    ("model", row.Model.DisplayName),
                    ("size", (row.VisionBytesRemaining / 1e9).ToString("0.0", Loc.Culture)),
                    ("setting", Loc.T("settings.downloads.cellular.title"))),
                Loc.T("common.ok"));
            return;
        }

        row.BeginVisionDownload();
        _app.Downloads.Start(row.Model, new[] { CatalogFileRole.Projector });
    }

    /// <summary>Add a draft independently, including to a text-only model or one whose
    /// projector was already downloaded. The global optional-file preference applies
    /// to the initial download; this button explicitly chooses this companion.</summary>
    private async void OnDraftAction(ModelRow? row)
    {
        if (row is null || !row.CanAddDraft)
            return;

        AppSettings settings = _app.Settings.Load();
        if (!settings.AllowCellularDownloads && Services.DeviceState.IsOnCellularOnly())
        {
            await DisplayAlert(
                Loc.T("models.alert.cellular.title"),
                Loc.T("models.alert.cellular.draft",
                    ("model", row.Model.DisplayName),
                    ("size", (row.DraftBytesRemaining / 1e9).ToString("0.0", Loc.Culture)),
                    ("setting", Loc.T("settings.downloads.cellular.title"))),
                Loc.T("common.ok"));
            return;
        }

        row.BeginDownload();
        _app.Downloads.Start(row.Model, new[] { CatalogFileRole.Draft });
    }

    /// <summary>
    /// Use this model now, and go back to the chat.
    ///
    /// <para>
    /// This used to save the choice and say it would apply "when TensorAgent next
    /// starts", which on a phone reads as a button that did nothing: the list said the
    /// model was selected while the chat kept answering "No model is configured".
    /// AgentAppHost.UseModel repoints the engine and loads the weights, so the choice
    /// is real by the time this returns.
    /// </para>
    /// <para>
    /// Loading is seconds of work (22 s for a 5 GB model on an iPhone 17 Pro Max), so
    /// it happens off the UI thread with the row showing what it is doing, and the page
    /// returns to the chat by itself afterwards -- being left on the list, having just
    /// chosen something, is a dead end the user has to navigate out of.
    /// </para>
    /// </summary>
    private async void Select(ModelRow row)
    {
        if (_selecting || !row.Runnable || !row.IsInstalled) return;
        _selecting = true;
        _loadingModelId = row.Model.Id;
        foreach (ModelRow item in _rows) item.SelectionInProgress = true;
        _search.Unfocus();
        // Choosing any model supersedes a promise to auto-select a different download
        // that happens to finish while this load is in flight.
        _pendingAutoSelectId = null;
        row.BeginLoading();
        try
        {
            string backend = await Task.Run(() => _app.UseModel(row.Model));
            _loadingModelId = null;
            Refresh();
            // Only if this is still the screen the user is looking at. Loading takes
            // twenty seconds and nobody is made to wait here for it: they can go back to
            // the chat, open the drawer and pick another screen while it runs. Popping
            // unconditionally when the load lands takes that screen away again, and from
            // the user's side it looks exactly like a menu item that did nothing.
            if (AppShell.IsOnTop(this))
                await AppShell.BackToChatAsync();
            Console.WriteLine($"TensorAgent: now using {row.Model.Id} on {backend}");
        }
        catch (Exception ex)
        {
            _loadingModelId = null;
            row.Failed(_app.Models, ex.Message);
            await DisplayAlert(Loc.T("models.alert.useFailed.title"), ex.Message, Loc.T("common.ok"));
            Refresh();
        }
        finally
        {
            _selecting = false;
            _loadingModelId = null;
            foreach (ModelRow item in _rows) item.SelectionInProgress = false;
        }
    }

    /// <summary>
    /// Hand the transfer to the download manager and let the row follow it.
    ///
    /// <para>
    /// Nothing is awaited here, and that is the change. This method used to hold the
    /// download for its whole length — the cancellation source lived in this page's
    /// dictionary and the progress went straight into a row — so a user who tapped
    /// Download and went back to the chat took the only owner of a multi-gigabyte
    /// transfer with them. The job is the app's now; this only starts it, and
    /// <see cref="OnDownloadChanged"/> paints whatever it goes on to do.
    /// </para>
    /// </summary>
    private void Download(ModelRow row)
    {
        _pendingAutoSelectId = row.Model.Id;
        row.BeginDownload();
        _app.Downloads.Start(
            row.Model,
            ModelDownloadManager.OptionalRolesFor(_app.Settings.Load().DownloadOptionalFiles));
    }

    private async void OnDelete(ModelRow? row)
    {
        if (row is null || !row.CanDelete)
            return;
        string message = row.Model.SideloadOnly
            ? Loc.T("models.alert.delete.messageLocal", ("model", row.Model.DisplayName))
            : Loc.T("models.alert.delete.message", ("model", row.Model.DisplayName));
        if (!await DisplayAlert(Loc.T("models.alert.delete.title"), message,
                Loc.T("models.alert.delete.confirm"), Loc.T("common.cancel")))
        {
            return;
        }
        _app.DeleteModel(row.Model);
        Refresh();
    }
}

internal sealed class BrowserTemplateSelector(DataTemplate model, DataTemplate family) : DataTemplateSelector
{
    protected override DataTemplate OnSelectTemplate(object item, BindableObject container) =>
        item is ModelFamilyRow ? family : model;
}

/// <summary>A disclosure row has its own identity so progress updates do not rebuild
/// the tree or reopen a family the user collapsed.</summary>
internal sealed class ModelFamilyRow(string id, string name) : BindableObject
{
    public string Id { get; } = id;
    public string Name { get; } = name;
    public string AutomationId => "ModelFamily-" + Id;
    public IReadOnlyList<ModelRow> Rows { get; private set; } = Array.Empty<ModelRow>();
    public bool IsExpanded { get; private set; }
    public string Title => (IsExpanded ? "▾  " : "▸  ") + Name;
    public string AccessibleLabel => Loc.T(IsExpanded ? "models.family.collapse" : "models.family.expand", ("family", Name));
    public string Summary => Loc.Plural("models.family.summary", Rows.Count,
        ("installed", Rows.Count(r => r.IsInstalled)), ("compatible", Rows.Count(r => r.Runnable)));
    public bool IsDownloading => _downloads > 0;
    public string Activity => Loc.T("models.family.downloading", ("count", _downloads));
    private int _downloads;

    public void Update(IReadOnlyList<ModelRow> rows, bool expanded)
    {
        Rows = rows;
        IsExpanded = expanded;
        OnPropertyChanged(nameof(Title));
        OnPropertyChanged(nameof(AccessibleLabel));
        OnPropertyChanged(nameof(Summary));
        UpdateActivity();
    }

    public void UpdateActivity()
    {
        int downloads = Rows.Count(r => r.IsDownloading);
        if (_downloads == downloads) return;
        _downloads = downloads;
        OnPropertyChanged(nameof(IsDownloading));
        OnPropertyChanged(nameof(Activity));
    }
}

/// <summary>One row of the model list, and the only place its display state lives.</summary>
public sealed class ModelRow : BindableObject
{
    private string _status;
    private double _fraction;
    private bool _busy;
    private bool _downloading;
    private string _actionLabel;
    private bool _selectionInProgress;

    /// <param name="download">
    /// What this launch's download manager is doing with the entry, or null when it has
    /// never touched it. It is a constructor parameter because the page is rebuilt every
    /// time it appears, and a row built without it shows "Partly downloaded · 3.1 GB
    /// still to fetch" beside a Download button for a transfer that is running right
    /// now — the one state the user must not be invited to start again.
    /// </param>
    public ModelRow(
        CatalogModel model, ModelStore store, string? selectedId, int deviceMemoryGB,
        ModelDownloadStatus? download = null, string? loadedModelName = null,
        bool loadedVisionReady = false, bool loadedDraftReady = false)
    {
        Model = model;
        Runnable = model.MinDeviceMemoryGB <= deviceMemoryGB;
        DeviceMemoryGB = deviceMemoryGB;
        RefreshInstallState(store);
        IsSelected = string.Equals(model.Id, selectedId, StringComparison.Ordinal);
        VisionActivationRequired = IsSelected
            && IsInstalled
            && model.Modalities.HasFlag(CatalogModalities.Image)
            && store.CompanionPath(model, CatalogFileRole.Projector) is not null
            && string.Equals(model.Weights.FileName, loadedModelName, StringComparison.OrdinalIgnoreCase)
            && !loadedVisionReady;
        DraftActivationRequired = IsSelected
            && IsInstalled
            && store.CompanionPath(model, CatalogFileRole.Draft) is not null
            && string.Equals(model.Weights.FileName, loadedModelName, StringComparison.OrdinalIgnoreCase)
            && !loadedDraftReady;
        _status = DescribeState(store);
        _actionLabel = !Runnable ? Loc.T("models.action.tooBig")
            : VisionActivationRequired ? Loc.T("models.action.enableVision")
            : DraftActivationRequired ? Loc.T("models.action.loadDraft")
            : IsInstalled ? (IsSelected ? Loc.T("models.action.selected") : Loc.T("models.action.use"))
            : model.SideloadOnly ? Loc.T("models.action.import")
            : Loc.T("models.action.download");

        if (download is { IsRunning: true } running)
        {
            BeginDownload();
            Report(running.Progress);
        }
        else if (download is { State: DownloadState.Failed } failed)
        {
            Failed(store, failed.Error ?? Loc.T("models.status.downloadFailed"));
        }
        else if (download is { State: DownloadState.Cancelled } && !IsInstalled)
        {
            Cancelled(store);
        }
    }

    public CatalogModel Model { get; }
    public string AutomationId => "Model-" + Model.Id;
    public string ActionAutomationId => "ModelAction-" + Model.Id;
    public Brush CardStroke => IsSelected ? new SolidColorBrush(Theme.Accent) : Brush.Transparent;
    public string FamilyAndMemory => ModelBrowser.FamilyName(Model) + " · "
        + Loc.T("models.row.requiredMemory", ("memory", Model.MinDeviceMemoryGB));
    public bool SelectionInProgress
    {
        get => _selectionInProgress;
        set
        {
            _selectionInProgress = value;
            OnPropertyChanged(nameof(CanAct));
            OnPropertyChanged(nameof(CanDelete));
            OnPropertyChanged(nameof(CanAddVision));
            OnPropertyChanged(nameof(CanAddDraft));
        }
    }
    public bool CanAct => Runnable && (IsDownloading || (!SelectionInProgress && !IsBusy));
    public bool IsDownloading
    {
        get => _downloading;
        private set { _downloading = value; OnPropertyChanged(); OnPropertyChanged(nameof(CanAct)); }
    }

    /// <summary>Whether this device has the memory the entry asks for.</summary>
    public bool Runnable { get; }

    public int DeviceMemoryGB { get; }
    public bool IsInstalled { get; private set; }
    public bool IsSelected { get; }
    /// <summary>Weights can answer text, but this advertised image model lacks its optional projector.</summary>
    public bool NeedsVisionProjector { get; private set; }
    /// <summary>Bytes left in the projector download, accounting for a resumable .part.</summary>
    public long VisionBytesRemaining { get; private set; }
    /// <summary>The projector arrived after this selected model was loaded text-only.</summary>
    public bool VisionActivationRequired { get; }
    public bool NeedsDraft { get; private set; }
    public long DraftBytesRemaining { get; private set; }
    public bool DraftActivationRequired { get; }
    public string AddDraftLabel => Loc.T("models.action.addDraft", ("size", Gb(DraftBytesRemaining)));

    public string Title => IsSelected ? Loc.T("models.row.inUse", ("model", Model.DisplayName)) : Model.DisplayName;

    /// <summary>
    /// What the model IS, in the order someone deciding actually asks: how big is it,
    /// what can it take in, and what is it for. The description used to appear only on
    /// experimental entries, so most of the list was a size and nothing else.
    /// </summary>
    public string Subtitle =>
        $"{Model.Parameters} · {Model.Quantization} · "
        + (Model.SideloadOnly
            ? Loc.T("models.row.localFileSize", ("size", Gb(Model.TotalBytes)))
            : Loc.T("models.row.downloadSize", ("size", Gb(Model.TotalBytes))))
        + (Model.Kind == CatalogArchitectureKind.MixtureOfExperts ? " · " + Loc.T("models.row.mixtureOfExperts") : string.Empty)
        + "\n" + Reads
        + (Model.SupportsThinking ? " · " + Loc.T("models.row.thinks") : string.Empty)
        + (string.IsNullOrWhiteSpace(Model.Notes) ? string.Empty
            : "\n" + (Model.Experimental ? Loc.T("models.row.experimental", ("notes", Model.Notes)) : Model.Notes));

    /// <summary>Every input this entry accepts, not just images, and for a model that makes
    /// pictures or clips rather than text, what it makes.</summary>
    private string Reads
    {
        get
        {
            var parts = new List<string> { Loc.T("models.row.input.text") };
            if (Model.Modalities.HasFlag(CatalogModalities.Image)) parts.Add(Loc.T("models.row.input.images"));
            if (Model.Modalities.HasFlag(CatalogModalities.Audio)) parts.Add(Loc.T("models.row.input.audio"));
            if (Model.Modalities.HasFlag(CatalogModalities.Video)) parts.Add(Loc.T("models.row.input.video"));
            string separator = Loc.T("models.row.input.separator");
            if ((Model.Modalities & (CatalogModalities.ImageOutput | CatalogModalities.VideoOutput)) == 0)
                return Loc.T("models.row.reads", ("inputs", string.Join(separator, parts)));
            // What it reads and what it makes is one sentence, so each kind is a whole line,
            // with and without inputs beyond the prompt.
            string inputs = string.Join(separator, parts.Skip(1));
            if (!Model.IsVideoGenerator)
            {
                return parts.Count == 1
                    ? Loc.T("models.row.makes.pictures")
                    : Loc.T("models.row.makes.picturesWithInputs", ("inputs", inputs));
            }
            if (Model.Modalities.HasFlag(CatalogModalities.AudioOutput))
            {
                return parts.Count == 1
                    ? Loc.T("models.row.makes.clipsWithSound")
                    : Loc.T("models.row.makes.clipsWithSoundWithInputs", ("inputs", inputs));
            }
            return parts.Count == 1
                ? Loc.T("models.row.makes.clips")
                : Loc.T("models.row.makes.clipsWithInputs", ("inputs", inputs));
        }
    }

    /// <summary>Greyed out when the device cannot run it, so the button reads as inert.</summary>
    public Color ActionColor => Runnable ? Theme.Accent : Theme.Surface;

    public string Status { get => _status; private set { _status = value; OnPropertyChanged(); } }
    public double Fraction { get => _fraction; private set { _fraction = value; OnPropertyChanged(); } }
    public bool IsBusy { get => _busy; private set { _busy = value; OnPropertyChanged(); OnPropertyChanged(nameof(CanAct)); OnPropertyChanged(nameof(CanDelete)); OnPropertyChanged(nameof(CanAddVision)); OnPropertyChanged(nameof(CanAddDraft)); } }
    public string ActionLabel { get => _actionLabel; private set { _actionLabel = value; OnPropertyChanged(); OnPropertyChanged(nameof(ActionDescription)); } }
    public string ActionDescription => ActionLabel + ": " + Model.DisplayName + " · " + Model.Quantization;
    public bool CanDelete => IsInstalled && !IsBusy && !SelectionInProgress;
    public bool CanAddVision => Runnable && NeedsVisionProjector && !IsBusy && !SelectionInProgress;
    public bool CanAddDraft => Runnable && NeedsDraft && !IsBusy && !SelectionInProgress;

    public void BeginDownload()
    {
        IsDownloading = true;
        IsBusy = true;
        ActionLabel = Loc.T("models.action.stop");
        Status = Loc.T("models.status.starting");
    }

    public void BeginVisionDownload()
    {
        IsDownloading = true;
        IsBusy = true;
        ActionLabel = Loc.T("models.action.stop");
        Status = Loc.T("models.status.startingVision");
    }

    public void BeginImport()
    {
        IsDownloading = false;
        IsBusy = true;
        ActionLabel = Loc.T("models.action.importing");
        Status = Loc.T("models.status.importing");
    }

    public void ReportImport(long bytes)
    {
        Fraction = Model.TotalBytes > 0 ? Math.Min(1.0, (double)bytes / Model.TotalBytes) : 0;
        Status = Loc.T("models.status.importProgress", ("copied", Gb(bytes)), ("total", Gb(Model.TotalBytes)));
    }

    /// <summary>Loading the weights, which is seconds rather than instant.</summary>
    public void BeginLoading()
    {
        IsDownloading = false;
        IsBusy = true;
        ActionLabel = Loc.T("models.action.loading");
        Status = Loc.T("models.status.loading");
    }

    public void Report(ModelDownloadProgress p)
    {
        Fraction = p.Fraction;
        // FileIndex is already 1-based (ModelDownloadProgress); adding one more showed the
        // first of seven files as "2/7" and a finished download as "8/7".
        var parts = new List<string>
        {
            p.Phase == "verifying"
                ? Loc.T("models.progress.verifying", ("file", p.FileIndex), ("files", p.FileCount))
                : Loc.T("models.progress.downloading", ("file", p.FileIndex), ("files", p.FileCount)),
            Loc.T("models.progress.size", ("received", Gb(p.BytesReceived)), ("total", Gb(p.TotalBytes))),
        };
        if (p.BytesPerSecond > 1)
            parts.Add(Loc.T("models.progress.speed", ("speed", (p.BytesPerSecond / 1e6).ToString("0.0", Loc.Culture))));
        if (p.Eta is { } left)
            parts.Add(Loc.T("models.progress.eta", ("minutes", Math.Round(left.TotalMinutes))));
        Status = string.Join(" · ", parts);
    }

    public void Finish(ModelStore store)
    {
        IsDownloading = false;
        IsBusy = false;
        RefreshInstallState(store);
        Fraction = 1;
        ActionLabel = InstalledActionLabel;
        Status = DescribeState(store);
        OnPropertyChanged(nameof(CanDelete));
        OnPropertyChanged(nameof(CanAddVision));
        OnPropertyChanged(nameof(CanAddDraft));
    }

    public void Cancelled(ModelStore store)
    {
        IsDownloading = false;
        IsBusy = false;
        RefreshInstallState(store);
        ActionLabel = IsInstalled
            ? InstalledActionLabel
            : Loc.T("models.action.resume");
        Status = Loc.T("models.status.stopped", ("state", DescribeState(store)));
        OnPropertyChanged(nameof(CanAddVision));
        OnPropertyChanged(nameof(CanAddDraft));
    }

    public void Failed(ModelStore store, string message)
    {
        IsDownloading = false;
        IsBusy = false;
        RefreshInstallState(store);
        ActionLabel = IsInstalled
            ? InstalledActionLabel
            : Model.SideloadOnly ? Loc.T("models.action.import")
            : Loc.T("models.action.retry");
        Status = message;
        OnPropertyChanged(nameof(CanAddVision));
        OnPropertyChanged(nameof(CanAddDraft));
    }

    private string InstalledActionLabel => VisionActivationRequired ? Loc.T("models.action.enableVision")
        : DraftActivationRequired ? Loc.T("models.action.loadDraft")
        : IsSelected ? Loc.T("models.action.selected") : Loc.T("models.action.use");

    private string DescribeState(ModelStore store)
    {
        if (!Runnable)
        {
            // Said as a fact about the hardware rather than as a refusal, and it names
            // both numbers so the user can see how far off it is instead of guessing.
            return Loc.T("models.status.needsDevice", ("required", Model.MinDeviceMemoryGB), ("available", DeviceMemoryGB));
        }

        if (NeedsVisionProjector)
        {
            return Loc.T("models.status.visionMissing", ("size", Gb(VisionBytesRemaining)), ("license", Model.License));
        }

        if (VisionActivationRequired)
        {
            return Loc.T("models.status.visionDownloaded", ("action", Loc.T("models.action.enableVision")), ("license", Model.License));
        }

        if (DraftActivationRequired)
            return Loc.T("models.status.draftDownloaded", ("action", Loc.T("models.action.loadDraft")), ("license", Model.License));

        if (Model.SideloadOnly && store.StateOf(Model) != InstallState.Installed)
        {
            return Loc.T("models.status.notImported", ("file", Model.Weights.FileName), ("license", Model.License));
        }

        return store.StateOf(Model) switch
        {
            InstallState.Installed => Loc.T("models.status.installed", ("size", Gb(store.InstalledBytes(Model))), ("license", Model.License)),
            InstallState.Partial => Loc.T("models.status.partial", ("size", Gb(store.RemainingBytes(Model)))),
            // What the download will actually move: files another installed entry already
            // holds are linked, not fetched (the second MiniMax-H3 checkpoint is its 11 GB
            // denoiser, not 35 GB).
            _ when store.RemainingBytes(Model) is long fetch && fetch < Model.TotalBytes =>
                Loc.T("models.status.notDownloadedShared", ("size", Gb(fetch)), ("license", Model.License)),
            _ => Loc.T("models.status.notDownloaded", ("size", Gb(Model.TotalBytes)), ("license", Model.License)),
        };
    }

    private void RefreshInstallState(ModelStore store)
    {
        IsInstalled = store.StateOf(Model) == InstallState.Installed;
        CatalogFile? projector = Model.Projector;
        NeedsVisionProjector = IsInstalled
            && projector is { Optional: true }
            && Model.Modalities.HasFlag(CatalogModalities.Image)
            && store.CompanionPath(Model, CatalogFileRole.Projector) is null;

        VisionBytesRemaining = NeedsVisionProjector
            ? store.RemainingBytes(Model, new[] { CatalogFileRole.Projector }) : 0;
        NeedsDraft = IsInstalled
            && Model.Files.Any(f => f.Role == CatalogFileRole.Draft && f.Optional)
            && store.CompanionPath(Model, CatalogFileRole.Draft) is null;
        DraftBytesRemaining = NeedsDraft
            ? store.RemainingBytes(Model, new[] { CatalogFileRole.Draft }) : 0;
        OnPropertyChanged(nameof(AddDraftLabel));
    }

    private static string Gb(long bytes) => (bytes / 1e9).ToString("0.00", Loc.Culture);
}

/// <summary>The Web UI's own palette, so the native pages do not look bolted on.</summary>
internal static class Theme
{
    public static readonly Color Background = Color.FromArgb("#0b1220");
    public static readonly Color Surface = Color.FromArgb("#151d31");
    public static readonly Color Text = Color.FromArgb("#e6edf7");
    public static readonly Color Muted = Color.FromArgb("#8b9ab8");
    public static readonly Color Accent = Color.FromArgb("#3b82f6");
    public static readonly Color Danger = Color.FromArgb("#ef4444");
}
