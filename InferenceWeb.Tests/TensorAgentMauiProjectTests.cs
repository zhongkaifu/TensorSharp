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
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Text.RegularExpressions;
using System.Xml.Linq;
using Xunit;

namespace InferenceWeb.Tests;

/// <summary>
/// Drift guards for the TensorAgent iOS head (TensorAgent/src/TensorAgent.Maui).
///
/// <para>
/// The head cannot be referenced from a net10.0 test project (it targets
/// net10.0-ios and needs the maui-ios workload), and the facts that make it
/// work are all declarative: the static NativeReference to the GgmlOps
/// xcframework with the exported_symbol linker flags, the phone-specific Web UI
/// bundled from the app's own wwwroot, the loopback ATS exception,
/// and the device-only memory entitlements. Each of these was hit for real
/// while bringing the app up, and each fails silently when removed - the
/// project still builds, and the app then dies at first P/Invoke or shows a
/// blank WebView. So the tests read the project files as XML and pin them.
/// </para>
/// </summary>
public class TensorAgentMauiProjectTests
{
    private static readonly XNamespace Ns = XNamespace.None;

    private static string RepoRoot
    {
        get
        {
            var dir = new DirectoryInfo(AppContext.BaseDirectory);
            while (dir != null && !File.Exists(Path.Combine(dir.FullName, "TensorSharp.slnx")))
                dir = dir.Parent;
            Assert.NotNull(dir);
            return dir!.FullName;
        }
    }

    private static string MauiDir => Path.Combine(RepoRoot, "TensorAgent", "src", "TensorAgent.Maui");

    private static string ScriptsDir => Path.Combine(RepoRoot, "TensorAgent", "scripts");

    private static string ShareExtensionDir =>
        Path.Combine(RepoRoot, "TensorAgent", "src", "TensorAgent.ShareExtension");

    private static string SharingDir =>
        Path.Combine(RepoRoot, "TensorAgent", "src", "TensorAgent.Sharing");

    private static XDocument Csproj => XDocument.Load(Path.Combine(MauiDir, "TensorAgent.Maui.csproj"));

    private static XDocument ShareExtensionCsproj =>
        XDocument.Load(Path.Combine(ShareExtensionDir, "TensorAgent.ShareExtension.csproj"));

    private static XDocument SharingCsproj =>
        XDocument.Load(Path.Combine(SharingDir, "TensorAgent.Sharing.csproj"));

    private static string? Property(XDocument doc, string name) =>
        doc.Descendants(Ns + name).Select(e => e.Value.Trim()).FirstOrDefault();

    [Fact]
    public void Solution_ListsTheMauiHead()
    {
        XDocument slnx = XDocument.Load(Path.Combine(RepoRoot, "TensorAgent", "TensorAgent.slnx"));
        Assert.Contains(
            slnx.Descendants("Project"),
            p => p.Attribute("Path")?.Value.Replace('\\', '/') == "src/TensorAgent.Maui/TensorAgent.Maui.csproj");
    }

    [Fact]
    public void DeviceDeploy_DefaultsToDebugAndInstallsTheConfigurationItBuilds()
    {
        string deploy = File.ReadAllText(Path.Combine(ScriptsDir, "deploy-device.sh"));
        string build = File.ReadAllText(Path.Combine(ScriptsDir, "build-device.sh"));
        const string appPath =
            "APP=\"${REPO_ROOT}/TensorAgent/src/TensorAgent.Maui/bin/${CONFIGURATION}/net10.0-ios/ios-arm64/TensorAgent.Maui.app\"";

        Assert.Contains("CONFIGURATION=\"${CONFIGURATION:-Debug}\"", deploy, StringComparison.Ordinal);
        Assert.Contains("CONFIGURATION=\"${CONFIGURATION:-Debug}\"", build, StringComparison.Ordinal);
        Assert.Contains(appPath, deploy, StringComparison.Ordinal);
        Assert.Contains(appPath, build, StringComparison.Ordinal);
        Assert.DoesNotContain(
            "TensorAgent.Maui/bin/Release/net10.0-ios/ios-arm64/TensorAgent.Maui.app",
            deploy,
            StringComparison.Ordinal);

        // The same validated value must reach build-device.sh; otherwise deploy-device.sh
        // can inspect or install a stale bundle from a different configuration.
        Assert.Contains("Debug|Release) ;;", deploy, StringComparison.Ordinal);
        Assert.Contains("CONFIGURATION=\"${CONFIGURATION}\" \\", deploy, StringComparison.Ordinal);
        Assert.Contains("CLEAN=1 \\", deploy, StringComparison.Ordinal);
        Assert.Contains("bash \"${SCRIPT_DIR}/build-device.sh\"", deploy, StringComparison.Ordinal);
        Assert.Contains("-c \"${CONFIGURATION}\"", build, StringComparison.Ordinal);
        Assert.Contains("if [[ \"${CLEAN:-0}\" == \"1\" ]]; then", build, StringComparison.Ordinal);
        Assert.Contains("dotnet restore \"${REPO_ROOT}/TensorAgent/src/TensorAgent.Maui/TensorAgent.Maui.csproj\"", build,
            StringComparison.Ordinal);
        Assert.Contains("-r ios-arm64", build, StringComparison.Ordinal);
        Assert.True(
            build.IndexOf("dotnet restore \"${REPO_ROOT}/TensorAgent/src/TensorAgent.Maui/TensorAgent.Maui.csproj\"", StringComparison.Ordinal) <
            build.IndexOf("dotnet clean \"${ARGS[@]}\"", StringComparison.Ordinal),
            "The device RID assets must be restored before the targeted clean resolves them.");
        Assert.Contains("-m:1", build, StringComparison.Ordinal);
        Assert.Contains("dotnet clean \"${ARGS[@]}\"", build, StringComparison.Ordinal);
        Assert.True(
            build.IndexOf("dotnet clean \"${ARGS[@]}\"", StringComparison.Ordinal) <
            build.IndexOf("dotnet build \"${ARGS[@]}\"", StringComparison.Ordinal),
            "The targeted device clean must finish before the build starts.");
        Assert.Contains("Interpreter=all", build, StringComparison.Ordinal);
        Assert.Contains("Registrar=static", build, StringComparison.Ordinal);
        Assert.Contains("xcrun devicectl --timeout \"${DEVICECTL_TIMEOUT}\" device install app", deploy,
            StringComparison.Ordinal);
        Assert.Contains("--device \"${DEVICE_ID}\" \"${APP}\"", deploy, StringComparison.Ordinal);
    }

    [Fact]
    public void SimulatorBuild_SerializesProjectsThatShareOutputFiles()
    {
        string script = File.ReadAllText(Path.Combine(ScriptsDir, "build-sim.sh"));

        Assert.Contains("-m:1", script, StringComparison.Ordinal);
        Assert.Contains("share one output directory", script, StringComparison.Ordinal);
    }

    [Fact]
    public void Head_TargetsIosWithTheAgreedIdentity()
    {
        XDocument doc = Csproj;
        Assert.Equal("net10.0-ios", Property(doc, "TargetFramework"));
        Assert.Equal("true", Property(doc, "UseMaui"));
        Assert.Equal("ai.tensorsharp.tensoragent", Property(doc, "ApplicationId"));
        Assert.Equal("TensorAgent", Property(doc, "ApplicationTitle"));
        // Must match build-ios.sh's deployment target or the static archive
        // refuses to link.
        Assert.Equal("17.0", Property(doc, "SupportedOSPlatformVersion"));
        Assert.Equal("partial", Property(doc, "TrimMode"));
        Assert.Equal("true", Property(doc, "JsonSerializerIsReflectionEnabledByDefault"));
        Assert.Equal("true", Property(doc, "TensorSharpSkipGgmlNative"));
        Assert.Equal("true", Property(doc, "TensorSharpSkipMlxNative"));
    }

    [Fact]
    public void Head_LinksGgmlOpsStaticallyAndExportsItsSymbols()
    {
        // The head links more than one framework now (CPython is embedded beside the
        // engine), so this picks out the engine's own reference rather than assuming
        // it is the only one.
        XElement native = Assert.Single(Csproj.Descendants(Ns + "NativeReference"),
            e => e.Attribute("Include")!.Value.Replace('\\', '/').Contains("GgmlOps.xcframework", StringComparison.Ordinal));
        Assert.EndsWith("TensorSharp.GGML.Native/build-ios/GgmlOps.xcframework", native.Attribute("Include")!.Value.Replace('\\', '/'));
        Assert.Equal("Static", native.Attribute("Kind")?.Value);
        Assert.Equal("True", native.Attribute("ForceLoad")?.Value);
        Assert.Equal("True", native.Attribute("IsCxx")?.Value);
        Assert.Equal("False", native.Attribute("SmartLink")?.Value);

        // GgmlNative.ImportResolver resolves DllImport("GgmlOps") to the main
        // program handle on iOS; dlsym only sees symbols the linker exported.
        string flags = native.Attribute("LinkerFlags")?.Value ?? string.Empty;
        Assert.Contains("-Wl,-exported_symbol,_TSGgml_*", flags);
        Assert.Contains("-Wl,-exported_symbol,_ggml_*", flags);
        Assert.Contains("-lc++", flags);

        string frameworks = native.Attribute("Frameworks")?.Value ?? string.Empty;
        foreach (string framework in new[] { "Foundation", "Metal", "MetalKit", "MetalPerformanceShaders", "MetalPerformanceShadersGraph", "Accelerate" })
            Assert.Contains(framework, frameworks.Split(' ', StringSplitOptions.RemoveEmptyEntries));
    }

    [Fact]
    public void Head_RetainsBonsaiNativeEntryPointsInRelease()
    {
        XDocument symbols = XDocument.Load(Path.Combine(MauiDir, "GgmlExportedSymbols.targets"));
        string[] retained = symbols.Descendants(Ns + "ReferenceNativeSymbol")
            .Select(e => e.Attribute("Include")?.Value)
            .OfType<string>()
            .ToArray();

        foreach (string export in new[]
                 {
                     "TSGgml_Qwen3ModelPrefill",
                     "TSGgml_Qwen3ModelDecodeLogits",
                     "TSGgml_Qwen3DropDecodeCache",
                     "TSGgml_Qwen3ResetDecodeCache",
                     "TSGgml_TransformerLayerDecode",
                     "TSGgml_TransformerModelDecode",
                     "TSGgml_Qwen35ArenaDiscardHostPointer",
                 })
        {
            Assert.Contains(export, retained);
        }

        // The failure is architectural rather than Bonsai-specific: an export
        // added to the static archive but omitted here survives Debug/simulator
        // builds and then disappears under the Release device strip step. Keep
        // the manifest identical to the native source exports so the next model
        // kernel cannot repeat that delayed EntryPointNotFound failure.
        string nativeDir = Path.Combine(RepoRoot, "TensorSharp.GGML.Native");
        var exportPattern = new Regex(
            @"^\s*TSG_EXPORT[^\r\n]*\b(TSGgml_[A-Za-z0-9_]+)\s*\(",
            RegexOptions.Multiline | RegexOptions.CultureInvariant);
        string[] nativeExports = Directory.EnumerateFiles(nativeDir, "*.*", SearchOption.TopDirectoryOnly)
            .Where(path => path.EndsWith(".cpp", StringComparison.Ordinal) ||
                           path.EndsWith(".c", StringComparison.Ordinal) ||
                           path.EndsWith(".inc", StringComparison.Ordinal) ||
                           path.EndsWith(".h", StringComparison.Ordinal))
            .SelectMany(path => exportPattern.Matches(File.ReadAllText(path))
                .Select(match => match.Groups[1].Value))
            .Distinct(StringComparer.Ordinal)
            .OrderBy(name => name, StringComparer.Ordinal)
            .ToArray();

        Assert.Equal(nativeExports,
            retained.Distinct(StringComparer.Ordinal)
                .OrderBy(name => name, StringComparer.Ordinal)
                .ToArray());
    }

    [Fact]
    public void Head_ReferencesAndRootsEveryEngineAssembly()
    {
        XDocument doc = Csproj;
        string[] engine =
        {
            "TensorSharp.Core",
            "TensorSharp.Runtime",
            "TensorSharp.Runtime.Logging",
            "TensorSharp.Models",
            "TensorSharp.Backends.GGML",
            "TensorSharp.AgentHost",
            "TensorSharp.Chat",
            "TensorAgent.Core",
        };

        var references = doc.Descendants(Ns + "ProjectReference")
            .Select(e => Path.GetFileNameWithoutExtension(e.Attribute("Include")!.Value.Replace('\\', '/')))
            .ToList();
        var roots = doc.Descendants(Ns + "TrimmerRootAssembly")
            .Select(e => e.Attribute("Include")!.Value)
            .ToList();

        foreach (string assembly in engine)
        {
            Assert.Contains(assembly, references);
            Assert.Contains(assembly, roots);
        }

        // TensorSharp.Server is ASP.NET Core, which has no iOS runtime pack.
        Assert.DoesNotContain("TensorSharp.Server", references);
    }

    [Fact]
    public void Head_EmbedsTheShareExtensionWithTheSameAppGroupAndASeparateProfileInput()
    {
        XDocument head = Csproj;
        Assert.Equal("true", Property(head, "TensorAgentShareExtension"));

        XElement[] entitlementSelections = head.Descendants(Ns + "CodesignEntitlements").ToArray();
        XElement sharingSelection = Assert.Single(entitlementSelections,
            e => e.Value.Trim() == "Platforms/iOS/Entitlements.Share.plist");
        Assert.Contains("TensorAgentShareExtension", sharingSelection.Attribute("Condition")?.Value ?? string.Empty,
            StringComparison.Ordinal);
        string sharingEntitlementsPath = Path.Combine(MauiDir, sharingSelection.Value.Trim().Replace('/', Path.DirectorySeparatorChar));
        XDocument sharingEntitlements = XDocument.Load(sharingEntitlementsPath);
        XElement groupKey = Assert.Single(sharingEntitlements.Descendants("key"),
            e => e.Value == "com.apple.security.application-groups");
        XElement groupArray = Assert.IsType<XElement>(groupKey.ElementsAfterSelf().First());
        Assert.Equal("array", groupArray.Name.LocalName);
        Assert.Contains(groupArray.Elements("string"),
            e => e.Value == "group.ai.tensorsharp.tensoragent");

        XElement reference = Assert.Single(head.Descendants(Ns + "ProjectReference"),
            e => e.Attribute("Include")?.Value.Replace('\\', '/').EndsWith(
                "TensorAgent.ShareExtension/TensorAgent.ShareExtension.csproj", StringComparison.Ordinal) == true);
        Assert.Equal("true", reference.Element(Ns + "IsAppExtension")?.Value.Trim());
        Assert.Contains("CodesignProvision=$(TensorAgentShareProvision)",
            reference.Element(Ns + "AdditionalProperties")?.Value ?? string.Empty,
            StringComparison.Ordinal);
        Assert.Contains("TensorAgentShareExtension",
            reference.Parent?.Attribute("Condition")?.Value ?? string.Empty,
            StringComparison.Ordinal);

        XElement headGroup = Assert.Single(head.Descendants(Ns + "CustomEntitlements"),
            e => e.Attribute("Include")?.Value == "com.apple.security.application-groups");
        XElement extensionGroup = Assert.Single(ShareExtensionCsproj.Descendants(Ns + "CustomEntitlements"),
            e => e.Attribute("Include")?.Value == "com.apple.security.application-groups");
        Assert.Equal("StringArray", headGroup.Attribute("Type")?.Value);
        Assert.Equal("StringArray", extensionGroup.Attribute("Type")?.Value);
        Assert.Equal("group.ai.tensorsharp.tensoragent", headGroup.Attribute("Value")?.Value);
        Assert.Equal(headGroup.Attribute("Value")?.Value, extensionGroup.Attribute("Value")?.Value);
    }

    [Fact]
    public void ShareExtension_MatchesTheContainingAppAndKeepsAMinimalDependencyGraph()
    {
        XDocument head = Csproj;
        XDocument extension = ShareExtensionCsproj;

        Assert.Equal("net10.0-ios", Property(extension, "TargetFramework"));
        Assert.Equal("Library", Property(extension, "OutputType"));
        Assert.Equal("true", Property(extension, "IsAppExtension"));
        Assert.Equal("partial", Property(extension, "TrimMode"));
        // A full-AOT managed-static registrar callback crashed before ViewDidLoad on
        // physical iOS 26. Keep the verified extension-only startup path explicit.
        Assert.Equal("true", Property(extension, "UseInterpreter"));
        Assert.Equal("static", Property(extension, "Registrar"));
        Assert.Equal(Property(head, "ApplicationDisplayVersion"), Property(extension, "ApplicationDisplayVersion"));
        Assert.Equal(Property(head, "ApplicationVersion"), Property(extension, "ApplicationVersion"));
        Assert.Equal(Property(head, "SupportedOSPlatformVersion"), Property(extension, "SupportedOSPlatformVersion"));

        string appId = Property(head, "ApplicationId")!;
        string extensionId = Property(extension, "ApplicationId")!;
        Assert.StartsWith(appId + ".", extensionId);
        Assert.Equal("ai.tensorsharp.tensoragent.share", extensionId);

        string[] references = extension.Descendants(Ns + "ProjectReference")
            .Select(e => Path.GetFileNameWithoutExtension(e.Attribute("Include")!.Value.Replace('\\', '/')))
            .ToArray();
        Assert.Equal(new[] { "TensorAgent.Sharing" }, references);
        Assert.Empty(extension.Descendants(Ns + "PackageReference"));
        Assert.Contains(extension.Descendants(Ns + "TrimmerRootAssembly"),
            e => e.Attribute("Include")?.Value == "TensorAgent.Sharing");

        XElement preprocessing = Assert.Single(extension.Descendants(Ns + "BundleResource"),
            e => e.Attribute("Include")?.Value == "ExtensionPreprocessing.js");
        Assert.True(File.Exists(Path.Combine(ShareExtensionDir, preprocessing.Attribute("Include")!.Value)));
        XElement privacy = Assert.Single(extension.Descendants(Ns + "BundleResource"),
            e => e.Attribute("LogicalName")?.Value == "PrivacyInfo.xcprivacy");
        Assert.Contains("TensorAgent.Maui/Platforms/iOS/Resources/PrivacyInfo.xcprivacy",
            privacy.Attribute("Include")?.Value.Replace('\\', '/') ?? string.Empty,
            StringComparison.Ordinal);

        // The shared wire-format assembly must stay usable by both processes without
        // pulling the inference engine (or any package) into the extension process.
        XDocument sharing = SharingCsproj;
        Assert.Equal("net10.0", Property(sharing, "TargetFramework"));
        Assert.Empty(sharing.Descendants(Ns + "ProjectReference"));
        Assert.Empty(sharing.Descendants(Ns + "PackageReference"));
        Assert.Contains(head.Descendants(Ns + "ProjectReference"),
            e => Path.GetFileNameWithoutExtension(e.Attribute("Include")?.Value) == "TensorAgent.Sharing");
    }

    [Fact]
    public void ShareExtension_InfoPlistDeclaresTheShareSurfaceAndItsPrincipalClass()
    {
        string plist = File.ReadAllText(Path.Combine(ShareExtensionDir, "Info.plist"));
        Assert.Contains("<string>com.apple.share-services</string>", plist, StringComparison.Ordinal);
        Assert.Contains("<string>ShareViewController</string>", plist, StringComparison.Ordinal);
        Assert.Contains("<string>ExtensionPreprocessing</string>", plist, StringComparison.Ordinal);
        foreach (string key in new[]
                 {
                     "NSExtensionActivationSupportsAttachmentsWithMaxCount",
                     "NSExtensionActivationSupportsFileWithMaxCount",
                     "NSExtensionActivationSupportsImageWithMaxCount",
                     "NSExtensionActivationSupportsMovieWithMaxCount",
                     "NSExtensionActivationSupportsText",
                     "NSExtensionActivationSupportsWebURLWithMaxCount",
                     "NSExtensionActivationSupportsWebPageWithMaxCount",
                 })
        {
            Assert.Contains($"<key>{key}</key>", plist, StringComparison.Ordinal);
        }

        // Identity/version values come from the two csproj files, where the drift test
        // above can compare them. A duplicate here wins the plist merge and can make an
        // otherwise green build uninstallable.
        Assert.DoesNotContain("<key>CFBundleIdentifier</key>", plist, StringComparison.Ordinal);
        Assert.DoesNotContain("<key>CFBundleShortVersionString</key>", plist, StringComparison.Ordinal);
        Assert.DoesNotContain("<key>CFBundleVersion</key>", plist, StringComparison.Ordinal);
        Assert.DoesNotContain("<key>MinimumOSVersion</key>", plist, StringComparison.Ordinal);

        string controller = File.ReadAllText(Path.Combine(ShareExtensionDir, "ShareViewController.cs"));
        Assert.Contains("[Register(\"ShareViewController\")]", controller, StringComparison.Ordinal);
    }

    [Fact]
    public void ShareExtension_DecodesBinaryPropertyListUrlsBeforeRawUtf8()
    {
        // UIActivityViewController on a physical iOS 26 device can vend an NSURL's
        // public.url file representation as a binary plist containing either one
        // NSString or Foundation's three-part NSURL array.
        // Treating that file as UTF-8 puts the visible `bplist00` header and trailer
        // bytes around an otherwise valid address in the chat composer.
        string reader = File.ReadAllText(Path.Combine(ShareExtensionDir, "ShareItemReader.cs"));
        int begin = reader.IndexOf(
            "private static Task<BoundedLoadedText> LoadUrlBoundedAsync", StringComparison.Ordinal);
        int end = reader.IndexOf(
            "private static Task<BoundedLoadedText> LoadTextBoundedAsync", begin, StringComparison.Ordinal);
        Assert.True(begin >= 0 && end > begin);
        string urlReader = reader[begin..end];

        Assert.Contains("\"bplist00\"u8", urlReader, StringComparison.Ordinal);
        Assert.Contains("DecodeBinaryPropertyListString(path, allowUrl: true)", urlReader,
            StringComparison.Ordinal);
        Assert.Contains("MaxUrlRepresentationBytes", reader, StringComparison.Ordinal);
        Assert.Contains("BoundUrlString(decoded)", urlReader, StringComparison.Ordinal);

        int plistDecode = urlReader.IndexOf(
            "DecodeBinaryPropertyListString(path, allowUrl: true)", StringComparison.Ordinal);
        int rawUtf8Fallback = urlReader.IndexOf("new StreamReader", StringComparison.Ordinal);
        Assert.True(plistDecode >= 0 && rawUtf8Fallback > plistDecode,
            "The bounded plist decoder must run before the raw-UTF-8 URL fallback.");
    }

    [Fact]
    public void ShareExtension_SecurelyDecodesKeyedArchiveTextAndUrlsWithinBounds()
    {
        // UIActivityViewController on physical iOS 26 vends a value-backed NSString
        // as an NSKeyedArchiver binary plist. NSURL can be either that keyed shape or
        // a direct binary-plist NSString. Both must be decoded before any raw UTF-8
        // reader sees the archive header/object table/trailer.
        string reader = File.ReadAllText(Path.Combine(ShareExtensionDir, "ShareItemReader.cs"));
        int helperBegin = reader.IndexOf(
            "private static string? DecodeBinaryPropertyListString", StringComparison.Ordinal);
        int helperEnd = reader.IndexOf(
            "private static BoundedLoadedText BoundUrlString", helperBegin, StringComparison.Ordinal);
        Assert.True(helperBegin >= 0 && helperEnd > helperBegin);
        string helper = reader[helperBegin..helperEnd];

        Assert.Contains("PropertyListWithData", helper, StringComparison.Ordinal);
        Assert.Contains("propertyList is NSArray urlParts", helper, StringComparison.Ordinal);
        Assert.Contains("urlParts.Count == 3", helper, StringComparison.Ordinal);
        Assert.Contains("urlParts.GetItem<NSObject>(0) is NSString relativeString", helper,
            StringComparison.Ordinal);
        Assert.Contains("urlParts.GetItem<NSObject>(1) is NSString baseString", helper,
            StringComparison.Ordinal);
        Assert.Contains("urlParts.GetItem<NSObject>(2) is NSDictionary", helper,
            StringComparison.Ordinal);
        Assert.Contains("archive[\"$archiver\"]", helper, StringComparison.Ordinal);
        Assert.Contains("\"NSKeyedArchiver\"", helper, StringComparison.Ordinal);
        Assert.Contains("NSKeyedUnarchiver.GetUnarchivedObject", helper, StringComparison.Ordinal);
        Assert.Contains("typeof(NSString)", helper, StringComparison.Ordinal);
        Assert.Contains("typeof(NSUrl)", helper, StringComparison.Ordinal);
        Assert.DoesNotContain("NSKeyedUnarchiver.UnarchiveObject(", reader, StringComparison.Ordinal);

        int textBegin = reader.IndexOf(
            "private static Task<BoundedLoadedText> LoadTextBoundedAsync", StringComparison.Ordinal);
        Assert.True(textBegin >= 0);
        string textReader = reader[textBegin..];
        Assert.Contains("MaxArchivedTextRepresentationBytes", reader, StringComparison.Ordinal);
        Assert.Contains("DecodeBinaryPropertyListString(path, allowUrl: false)", textReader,
            StringComparison.Ordinal);
        Assert.Contains("string.Equals(typeIdentifier, Html", textReader, StringComparison.Ordinal);
        Assert.Contains("ShareText.FromHtml(decoded)", textReader, StringComparison.Ordinal);
        Assert.Contains("string.Equals(typeIdentifier, Rtf", textReader, StringComparison.Ordinal);
        Assert.Contains("NSData.FromString(decoded, NSStringEncoding.UTF8)", textReader,
            StringComparison.Ordinal);
        int keyedDecode = textReader.IndexOf(
            "DecodeBinaryPropertyListString(path, allowUrl: false)", StringComparison.Ordinal);
        int rtfDecode = textReader.IndexOf("bool isRtf", StringComparison.Ordinal);
        int rawUtf8 = textReader.IndexOf("new StreamReader", StringComparison.Ordinal);
        Assert.True(keyedDecode >= 0 && rtfDecode > keyedDecode && rawUtf8 > keyedDecode,
            "Keyed plain text must be bounded and decoded before rich/raw text fallbacks.");
    }

    [Fact]
    public void Head_BundlesItsOwnPhoneWwwroot()
    {
        // Several things are bundled now — the skills, the Python standard library —
        // so this is the one that matters: the phone-specific Web UI. It deliberately
        // differs from the desktop Server page while speaking the same loopback API.
        XElement bundle = Assert.Single(Csproj.Descendants(Ns + "BundleResource"),
            e => e.Attribute("Include")!.Value.Replace('\\', '/') == "wwwroot/**/*");
        Assert.StartsWith("webui/", bundle.Attribute("Link")?.Value.Replace('\\', '/'));

        string page = Path.Combine(MauiDir, "wwwroot", "index.html");
        Assert.True(File.Exists(page), "TensorAgent.Maui must carry its phone-specific index.html.");
        Assert.DoesNotContain(
            Csproj.Descendants(Ns + "BundleResource"),
            e => e.Attribute("Include")!.Value.Replace('\\', '/')
                .Contains("TensorSharp.Server/wwwroot", StringComparison.Ordinal));
    }

    [Fact]
    public void Head_BundlesExactlyTheSkillsTheVerifierPassed()
    {
        // TensorAgent/skills is shared with the desktop hosts, so it holds skills the
        // phone cannot run. Playwright and web-artifacts-builder were bundled for a while:
        // their shell scripts run npx, pnpm and npm, the app has no Node package manager,
        // and a skill the model is offered but cannot run fails in front of the user on
        // its first step. The csproj excludes such a skill by name because MSBuild cannot
        // read verdicts.json; this keeps the two equal.
        XElement bundle = Assert.Single(Csproj.Descendants(Ns + "BundleResource"),
            e => e.Attribute("Include")!.Value.Replace('\\', '/') == "../../skills/**/*");
        Assert.StartsWith("skills/", bundle.Attribute("Link")?.Value.Replace('\\', '/'));

        string[] excludes = (bundle.Attribute("Exclude")?.Value ?? string.Empty)
            .Split(';', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .Select(pattern => pattern.Replace('\\', '/'))
            .ToArray();
        Assert.Contains("../../skills/verdicts.json", excludes);
        Assert.Contains("../../skills/**/__pycache__/**/*", excludes);

        var wholeSkill = new Regex(@"^\.\./\.\./skills/([^/*]+)/\*\*/\*$", RegexOptions.CultureInvariant);
        string[] excluded = excludes
            .Select(pattern => wholeSkill.Match(pattern))
            .Where(match => match.Success)
            .Select(match => match.Groups[1].Value)
            .OrderBy(name => name, StringComparer.Ordinal)
            .ToArray();

        string skillsDir = Path.Combine(RepoRoot, "TensorAgent", "skills");
        string[] skills = Directory.EnumerateDirectories(skillsDir)
            .Where(dir => File.Exists(Path.Combine(dir, "SKILL.md")))
            .Select(dir => Path.GetFileName(dir))
            .OrderBy(name => name, StringComparer.Ordinal)
            .ToArray();

        using JsonDocument verdicts = JsonDocument.Parse(File.ReadAllText(Path.Combine(skillsDir, "verdicts.json")));
        string[] passed = verdicts.RootElement.EnumerateArray()
            .Where(verdict => verdict.GetProperty("ok").GetBoolean())
            .Select(verdict => verdict.GetProperty("name").GetString()!)
            .OrderBy(name => name, StringComparer.Ordinal)
            .ToArray();

        Assert.Contains("playwright", skills);
        Assert.Contains("playwright", excluded);
        // The same blind spot let web-artifacts-builder through: its scripts install and run
        // pnpm, npm and parcel, none of which the in-app shell has.
        Assert.Contains("web-artifacts-builder", excluded);

        var isSkill = new HashSet<string>(skills, StringComparer.Ordinal);
        var didPass = new HashSet<string>(passed, StringComparer.Ordinal);
        foreach (string name in excluded)
        {
            Assert.True(isSkill.Contains(name),
                $"The csproj excludes skills/{name}, which is not a skill directory any more; drop the stale exclusion.");
            Assert.False(didPass.Contains(name),
                $"verdicts.json says '{name}' works in the app, but the csproj keeps it out of the bundle.");
        }

        string[] bundled = skills.Except(excluded, StringComparer.Ordinal).ToArray();
        foreach (string name in bundled)
        {
            Assert.True(didPass.Contains(name),
                $"skills/{name} ships in the iOS bundle but verdicts.json has no passing verdict for it. " +
                "Run scripts/verify-skills.py against the staged runtime and record the verdict, or exclude " +
                "the directory in TensorAgent.Maui.csproj if it cannot run on a phone.");
        }

        // And the other direction: a skill the verifier passed must actually ship.
        Assert.Equal(passed, bundled);
    }

    [Fact]
    public void InfoPlist_AllowsLoopbackHttpAndDeclaresTheUsageStrings()
    {
        string plist = File.ReadAllText(Path.Combine(MauiDir, "Platforms", "iOS", "Info.plist"));
        // WKWebView -> http://127.0.0.1 is blocked by ATS without this.
        Assert.Contains("<key>NSAllowsLocalNetworking</key>", plist);
        foreach (string key in new[]
                 {
                     "NSCameraUsageDescription",
                     "NSMicrophoneUsageDescription",
                     "NSPhotoLibraryUsageDescription",
                     "NSSpeechRecognitionUsageDescription",
                     "UIFileSharingEnabled",
                     "LSSupportsOpeningDocumentsInPlace",
                 })
        {
            Assert.Contains($"<key>{key}</key>", plist);
        }
    }

    [Fact]
    public void Entitlements_AreDeviceOnlyAndRaiseTheMemoryLimit()
    {
        string entitlements = File.ReadAllText(Path.Combine(MauiDir, "Platforms", "iOS", "Entitlements.plist"));
        Assert.Contains("<key>com.apple.developer.kernel.increased-memory-limit</key>", entitlements);
        Assert.Contains("<key>com.apple.developer.kernel.extended-virtual-addressing</key>", entitlements);

        // The simulator has no jetsam limit and CompileEntitlements must not emit
        // codesign entitlements for simulator builds, so the file is wired up
        // for the device RID only.
        XElement[] codesign = Csproj.Descendants(Ns + "CodesignEntitlements").ToArray();
        Assert.Equal(2, codesign.Length);
        Assert.All(codesign, item =>
        {
            Assert.Contains("ios-arm64", item.Attribute("Condition")?.Value);
            Assert.DoesNotContain("simulator", item.Attribute("Condition")?.Value);
        });
    }
}
