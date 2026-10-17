// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

namespace TensorAgent.Maui;

/// <summary>
/// One place, and four screens that sit on top of it.
///
/// <para>
/// The chat is the shell's ONLY content. The saved chats, the model list, the settings
/// and the about page are pushed over it and popped off again, rather than being
/// sibling items the shell switches between — and that is not a matter of taste.
/// Switching top-level items tears the page down: the shell renderer disposes the
/// outgoing item's renderer, which disconnects the page's handler and removes its view
/// controller. Pushing does none of that; the chat stays in the stack with its handler
/// intact, so it comes back instantly and with its state.
/// </para>
/// <para>
/// What pushing does NOT buy, and no navigation shape here can, is keeping the WebView
/// running. UIKit takes a covered view controller's view out of the window, WebKit drops
/// the foreground assertion for a WKWebView whose window is nil, and the content process
/// is suspended — mid-answer. That is why the generation cannot belong to the page at
/// all: it belongs to <see cref="Core.Sessions.ChatTurnManager"/>, and the page attaches
/// to it again on the way back. This class only makes sure there is a page to come back
/// to.
/// </para>
/// <para>
/// The flyout is gone with it. The page has its own menu now — a drawer from the left
/// edge that also lists the saved chats — and two menus for the same five destinations
/// is one more than a phone screen can justify.
/// </para>
/// </summary>
public sealed class AppShell : Shell
{
    private readonly Dictionary<string, Page> _pages;

    /// <summary>One navigation at a time; see <see cref="OpenAsync"/>.</summary>
    private static readonly SemaphoreSlim Gate = new(1, 1);

    /// <summary>How long a pop is given before this gives up on it and pushes anyway.</summary>
    private static readonly TimeSpan PopTimeout = TimeSpan.FromSeconds(3);

    public AppShell(MainPage chat, Pages.SessionsPage sessions, Pages.ModelsPage models, Pages.SettingsPage settings, Pages.AboutPage about)
    {
        Title = "TensorAgent";
        FlyoutBehavior = FlyoutBehavior.Disabled;
        BackgroundColor = Pages.Theme.Background;

        var main = new ShellContent { Title = Core.Localization.Loc.T("shell.chat.title"), Route = "main", Content = chat };
        Items.Add(main);
        // Keep this hidden shell item stable after it is attached. MAUI still binds a
        // flyout cell even with FlyoutBehavior.Disabled; changing its title can reload
        // a cell whose IndexPath is null and abort UIKit with NSInvalidArgumentException.
        // The visible page titles and the chat's own menu localize independently.
        _pages = new Dictionary<string, Page>(StringComparer.Ordinal)
        {
            ["sessions"] = sessions,
            ["models"] = models,
            ["settings"] = settings,
            ["about"] = about,
        };

        // TENSORAGENT_USE_MODEL=<catalog id>: in a Debug build always (the simulator
        // harness cannot tap); in a Release build only when an on-device benchmark asked
        // for it (TENSORAGENT_SPEC_BENCH=1, TENSORAGENT_IMAGE_BENCH=1 for pictures, or
        // TENSORAGENT_VIDEO_BENCH=1 for clips), which is how scripts/bench-spec-device.sh
        // chooses the model to measure - a Release build used to ignore the variable and
        // measure whatever model was remembered.
        bool honourUseModel = Core.Hosting.SpeculationBench.Requested || Core.Hosting.ImageBench.Requested
            || Core.Hosting.VideoBench.Requested;
#if DEBUG
        honourUseModel = true;
#endif
        if (honourUseModel)
        {
            string? use = Environment.GetEnvironmentVariable("TENSORAGENT_USE_MODEL");
            if (!string.IsNullOrWhiteSpace(use))
            {
                Dispatcher.Dispatch(async () =>
                {
                    try
                    {
                        Core.Catalog.CatalogModel? picked = Core.Catalog.ModelCatalog.Find(use.Trim());
                        if (picked is null)
                        {
                            Console.WriteLine($"TensorAgent: TENSORAGENT_USE_MODEL={use} is not a catalog id");
                            return;
                        }
                        string backend = await Task.Run(() => models.Host.UseModel(picked));
                        Console.WriteLine($"TensorAgent: debug hook loaded {picked.Id} on {backend}");
                        await OpenAsync("main");
                    }
                    catch (Exception ex) { Console.WriteLine("TensorAgent: debug model use failed: " + ex.Message); }
                });
            }
        }

#if DEBUG
        // The simulator harness cannot tap: simctl has no way to touch the screen, so
        // without this only the first page could ever be screenshotted. Launching with
        // TENSORAGENT_START_PAGE=models opens that route instead, which is how
        // scripts/run-sim.sh captures the model list and the settings.
        // TENSORAGENT_USE_MODEL=<catalog id> reproduces, without a tap, exactly what a
        // user does in the Models list: choose an installed model, have it loaded, and
        // land back in the chat. It exists because the bug this guards against was
        // reported twice from a phone and could not be reproduced from the outside --
        // devicectl cannot touch the screen, so the one path that mattered was the one
        // path no test could drive. Paired with TENSORAGENT_DEMO_PROMPT it drives the
        // whole reported failure: select a model, then send a prompt, and see whether
        // the page still refuses with "No model is configured".
        string? start = Environment.GetEnvironmentVariable("TENSORAGENT_START_PAGE");
        if (!string.IsNullOrWhiteSpace(start))
            Dispatcher.Dispatch(async () => await OpenAsync(start.Trim()));
#endif
    }

    /// <summary>
    /// Show one of the app's other screens over the chat, or come back to the chat.
    ///
    /// <para>
    /// The stack is never deeper than one: whatever is already on top comes off first,
    /// so "Models" from a page reached through "Settings" leaves one page over the chat
    /// rather than three, and the back arrow always means "back to the chat". A route
    /// this shell does not have is ignored rather than guessed at, which is what stops a
    /// page from navigating the app somewhere it has no screen.
    /// </para>
    /// </summary>
    public static async Task OpenAsync(string? route)
    {
        if (Current is not AppShell shell)
        {
            Console.WriteLine($"TensorAgent: open {route} ignored, the shell is {Current?.GetType().Name ?? "null"}");
            return;
        }
        // Decided BEFORE anything is popped, so a route this shell does not have leaves
        // the app exactly where it was rather than quietly closing the page the user is
        // looking at.
        Page? page = null;
        bool chat = route is null or "main";
        if (!chat && !shell._pages.TryGetValue(route!, out page))
        {
            Console.WriteLine($"TensorAgent: open {route} ignored, this shell has no such route");
            return;
        }

        // One navigation at a time. Everything below reads the stack and then awaits, and
        // a value read before an await is a guess afterwards: the MAUI stack is mutated
        // several dispatcher hops after PushAsync is called, and iOS's Shell renderer
        // does not serialise navigations for us. Two callers inside here at once is not
        // hypothetical -- a second tap while the drawer slides shut, ModelsPage
        // returning to the chat when a model finishes loading, and UIKit's own back
        // button all arrive as independent main-thread work items -- and every way they
        // interleave ends with the user on the chat: the loser of a double pop throws
        // "Can't pop last page off stack", a pop's completion is a single overwritable
        // field that can be left pending forever, and a pop's late teardown removes the
        // view controller a concurrent re-push just installed.
        await Gate.WaitAsync().ConfigureAwait(true);
        try
        {
            INavigation navigation = shell.Navigation;

            // Asking for the page already on top is a no-op, not a pop and a push. This
            // is what makes an impatient double tap harmless.
            if (!chat && Top(navigation) is { } top && ReferenceEquals(top, page))
                return;

            // Shell's stack carries a null placeholder for the shell content itself, so
            // anything past the first entry is a page pushed over the chat. Bounded
            // rather than `while`: a pop that does not shorten the stack would otherwise
            // spin forever on the UI thread, and the stack is never deeper than one.
            for (int i = 0; i < 8 && navigation.NavigationStack.Count > 1; i++)
            {
                // Never awaited unbounded. The renderer resolves a pop through a single
                // field that a concurrent pop or an interactive back-swipe overwrites,
                // and an unresolved one leaves this method parked between the pop and
                // the push -- no exception, no log, and a menu item that did nothing.
                Task pop = navigation.PopAsync(false);
                if (await Task.WhenAny(pop, Task.Delay(PopTimeout)).ConfigureAwait(true) != pop)
                {
                    Console.WriteLine(
                        $"TensorAgent: open {route}: a pop did not complete in {PopTimeout.TotalSeconds:0.#}s "
                        + $"with {navigation.NavigationStack.Count} on the stack; pushing anyway");
                    break;
                }
                await pop.ConfigureAwait(true);
            }

            // The pages are DI singletons, so the same instance is pushed for the life of
            // the app. MAUI adds it to the stack a second time without complaint and then
            // hands UIKit a view controller that is already installed, which is a
            // corrupted stack rather than an error.
            if (!chat && !navigation.NavigationStack.Contains(page!))
                await navigation.PushAsync(page!).ConfigureAwait(true);

            Console.WriteLine(
                $"TensorAgent: open {route ?? "main"} -> stack {navigation.NavigationStack.Count}, "
                + $"top {Top(navigation)?.GetType().Name ?? "chat"}");
        }
        catch (Exception ex)
        {
            // The whole exception, not just the message: the type is what tells a
            // "Can't pop last page off stack" race apart from a page that threw in
            // OnAppearing, and both land the user on the chat looking identical.
            Console.WriteLine($"TensorAgent: open {route} failed: " + ex);
        }
        finally
        {
            Gate.Release();
        }
    }

    /// <summary>Come back to the chat, from a page that is done.</summary>
    public static Task BackToChatAsync() => OpenAsync("main");

    /// <summary>
    /// Whether <paramref name="page"/> is the page the user is looking at.
    ///
    /// <para>
    /// For deferred work that wants to return to the chat when it finishes. Loading a
    /// model takes twenty seconds, and the user does not have to wait on the Models
    /// screen while it happens -- so "go back to the chat now" has to mean "…if the
    /// screen I was started from is still the one on top", or it pops whatever the user
    /// opened in the meantime and their menu tap looks like it did nothing.
    /// </para>
    /// </summary>
    public static bool IsOnTop(Page page)
        => Current is AppShell shell && ReferenceEquals(Top(shell.Navigation), page);

    private static Page? Top(INavigation navigation)
    {
        IReadOnlyList<Page> stack = navigation.NavigationStack;
        return stack.Count > 1 ? stack[^1] : null;
    }
}
