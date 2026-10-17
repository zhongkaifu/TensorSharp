// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
namespace TensorSharp.Studio;

/// <summary>All manual and agent edits execute this same pure command interpreter against a detached project.
/// Validation of the whole resulting project happens before persistence, giving batches atomic semantics.</summary>
public static class StudioCommands
{
    public static void Apply(StudioProject project, IReadOnlyList<StudioCommand> commands, StudioOptions options)
    {
        if (commands is null || commands.Any(c => c is null) || commands.Count is < 1 or > 100) throw new StudioException("invalid_commands", "An edit transaction must contain between 1 and 100 commands.");
        foreach (var command in commands) { ValidateCommandFields(command); ApplyOne(project, command); }
        Validate(project, options);
    }

    public static void Validate(StudioProject p, StudioOptions options)
    {
        if (p.SchemaVersion != 1) throw new StudioException("unsupported_schema", $"Project schema {p.SchemaVersion} is not supported; this application supports schema 1.");
        StudioPaths.Id(p.Id);
        if (p.FrameRate is null || p.Assets is null || p.Tracks is null || p.Clips is null || p.Captions is null ||
            p.Assets.Any(a => a is null) || p.Tracks.Any(t => t is null) || p.Clips.Any(c => c is null) || p.Captions.Any(c => c is null))
            throw new StudioException("invalid_document", "Project settings and collections must not be null.");
        p.FrameRate.Validate();
        if (p.ConversationId is { } conversation && (conversation.Length > 200 || conversation.Any(char.IsControl)))
            throw new StudioException("invalid_conversation", "Conversation references must be at most 200 printable characters.");
        if (string.IsNullOrWhiteSpace(p.Name) || p.Name.Length > 200) throw new StudioException("invalid_name", "Project name must contain 1–200 characters.");
        if (p.Width is < 16 or > 4096 || p.Height is < 16 or > 4096 || p.Width % 2 != 0 || p.Height % 2 != 0)
            throw new StudioException("invalid_resolution", "Project dimensions must be even and between 16 and 4096 pixels.");
        if (p.Tracks.Count > 32 || p.Clips.Count > 2000 || p.Assets.Count > 2000 || p.Captions.Count > 5000)
            throw new StudioException("project_limit", "Project exceeds the supported track, clip, asset, or caption limit.");
        Unique(p.Tracks.Select(t => t.Id)); Unique(p.Clips.Select(c => c.Id)); Unique(p.Assets.Select(a => a.Id)); Unique(p.Captions.Select(c => c.Id));
        foreach (var track in p.Tracks)
        {
            if (track.Kind is not ("video" or "audio")) throw new StudioException("invalid_track", "Track kind must be video or audio.");
            Volume(track.Volume);
            if (string.IsNullOrWhiteSpace(track.Name) || track.Name.Length > 200) throw new StudioException("invalid_name", "Track name is too long.");
        }
        foreach (var asset in p.Assets)
        {
            StudioPaths.FileName(asset.FileName);
            if (asset.Generation is { } provenance && (provenance.Request is null || provenance.EffectiveParameters is null || provenance.ModelVersion is null))
                throw new StudioException("invalid_asset", "Generation provenance must include a request, model identity, and effective parameters.");
            if (asset.Bytes < 0 || asset.Bytes > options.MaxImportBytes || string.IsNullOrWhiteSpace(asset.Name) || asset.Name.Length > 200 || asset.Width < 0 || asset.Height < 0 || asset.Width > 16384 || asset.Height > 16384 ||
                asset.Sha256 is null || asset.Sha256.Length != 64 || !asset.Sha256.All(c => c is >= '0' and <= '9' or >= 'a' and <= 'f'))
                throw new StudioException("invalid_asset", "Asset sizes, dimensions, name, or content hash are invalid.");
            if (asset.ContentType != StudioMedia.ContentType(Path.GetExtension(asset.FileName)))
                throw new StudioException("invalid_asset", "Asset content type does not match its managed media extension.");
            if (asset.Kind is not ("video" or "image" or "audio") || !double.IsFinite(asset.DurationSeconds) || asset.DurationSeconds < 0)
                throw new StudioException("invalid_asset", "Asset metadata is invalid.");
        }
        foreach (var c in p.Clips)
        {
            var t = Track(p, c.TrackId); var a = Asset(p, c.AssetId);
            if (t.Kind == "audio" && !a.HasAudio) throw new StudioException("incompatible_track", "An audio track requires media with an audio stream.");
            if (t.Kind == "video" && a.Kind == "audio") throw new StudioException("incompatible_track", "Audio media must be placed on an audio track.");
            Range(c.StartFrame, c.DurationFrames, options);
            if (c.SourceStartFrame < 0 || c.SourceStartFrame > options.MaxTimelineFrames)
                throw new StudioException("invalid_trim", "Source start must be a nonnegative frame in the supported time range.");
            if (a.Kind != "image" && p.FrameRate.Seconds(checked(c.SourceStartFrame + c.DurationFrames)) > a.DurationSeconds + p.FrameRate.Seconds(1) + 0.0001)
                throw new StudioException("invalid_trim", "The clip extends beyond the source media duration.");
            Volume(c.Volume);
            if (c.FadeInFrames < 0 || c.FadeOutFrames < 0 || c.AudioFadeOutFrames < 0 || c.FadeInFrames > c.DurationFrames || Math.Max(c.FadeOutFrames, c.AudioFadeOutFrames) > c.DurationFrames - c.FadeInFrames)
                throw new StudioException("invalid_fade", "Fades must be nonnegative and fit within the clip duration.");
            if (c.Fit is not ("contain" or "cover")) throw new StudioException("invalid_fit", "Fit must be contain or cover.");
            if (c.LinkGroup != null) StudioPaths.Id(c.LinkGroup);
            if (c.ShotId != null && c.ShotId.Length > 200) throw new StudioException("invalid_shot", "Shot identifier exceeds 200 characters.");
        }
        foreach (var caption in p.Captions)
        {
            Range(caption.StartFrame, caption.DurationFrames, options);
            if (string.IsNullOrWhiteSpace(caption.Text) || caption.Text.Length > 2000 || caption.Text.Any(ch => char.IsControl(ch) && ch is not ('\n' or '\r' or '\t')))
                throw new StudioException("invalid_caption", "Caption must contain 1–2000 printable characters.");
        }
    }

    static void Unique(IEnumerable<string> ids)
    {
        var seen = new HashSet<string>(StringComparer.Ordinal);
        foreach (var id in ids) if (!seen.Add(StudioPaths.Id(id))) throw new StudioException("duplicate_id", "Project identifiers must be unique within their type.");
    }
    static void Volume(double value)
    {
        if (!double.IsFinite(value) || value is < 0 or > 4) throw new StudioException("invalid_volume", "Volume must be between 0 and 4.");
    }
    static void Range(long start, long duration, StudioOptions options)
    {
        if (start < 0 || duration <= 0 || start > options.MaxTimelineFrames || duration > options.MaxTimelineFrames - start)
            throw new StudioException("invalid_range", "Timeline ranges must have nonnegative starts and positive durations within the project limit.");
    }
    static StudioTrack Track(StudioProject p, string? id) => p.Tracks.FirstOrDefault(t => t.Id == id) ?? throw new StudioException("track_not_found", "Track does not exist in this project.");
    static StudioClip Clip(StudioProject p, string? id) => p.Clips.FirstOrDefault(c => c.Id == id) ?? throw new StudioException("clip_not_found", "Clip does not exist in this project.");
    static StudioAsset Asset(StudioProject p, string? id) => p.Assets.FirstOrDefault(a => a.Id == id) ?? throw new StudioException("asset_not_found", "Asset does not exist in this project.");
    static StudioCaption Caption(StudioProject p, string? id) => p.Captions.FirstOrDefault(c => c.Id == id) ?? throw new StudioException("caption_not_found", "Caption does not exist in this project.");
    static IEnumerable<StudioClip> Linked(StudioProject p, StudioClip clip, bool linked) => linked && !string.IsNullOrEmpty(clip.LinkGroup) ? p.Clips.Where(c => c.LinkGroup == clip.LinkGroup).ToArray() : [clip];
    static long Required(long? value, string field) => value ?? throw new StudioException("missing_parameter", $"{field} is required.");

    /// <summary>Every command type the interpreter accepts, with the only fields each may carry besides ApplyToLinked.
    /// It is the gate: a type missing here is refused even if ApplyOne has a case for it. Anything that describes the
    /// command set to a client (the agent's studio_propose declaration) is drift-tested against it.</summary>
    static readonly Dictionary<string, string[]> CommandFields = new(StringComparer.Ordinal)
    {
        ["addTrack"] = ["TrackId", "Name", "Kind"],
        ["updateTrack"] = ["TrackId", "Name", "Volume", "Muted", "Visible"],
        ["reorderTrack"] = ["TrackId", "Index"],
        ["removeTrack"] = ["TrackId"],
        ["insertClip"] = ["ClipId", "TrackId", "AssetId", "StartFrame", "SourceStartFrame", "DurationFrames", "Volume", "FadeInFrames", "FadeOutFrames", "AudioFadeOutFrames", "LinkGroup", "ShotId", "Fit"],
        ["moveClip"] = ["ClipId", "TrackId", "StartFrame"],
        ["trimClip"] = ["ClipId", "StartFrame", "SourceStartFrame", "DurationFrames"],
        ["updateClip"] = ["ClipId", "Volume", "FadeInFrames", "FadeOutFrames", "AudioFadeOutFrames", "Fit", "LinkGroup", "ShotId"],
        ["dissolve"] = ["ClipId", "OtherClipId"],
        ["splitClip"] = ["ClipId", "AtFrame"],
        ["duplicateClip"] = ["ClipId", "StartFrame"],
        ["removeClip"] = ["ClipId"],
        ["replaceAsset"] = ["ClipId", "AssetId"],
        ["deleteAsset"] = ["AssetId", "RemoveTimelineClips"],
        ["addCaption"] = ["CaptionId", "Text", "StartFrame", "DurationFrames"],
        ["updateCaption"] = ["CaptionId", "Text", "StartFrame", "DurationFrames"],
        ["removeCaption"] = ["CaptionId"],
        ["settings"] = ["Name", "Width", "Height", "ConversationId"],
    };

    public static IReadOnlyCollection<string> CommandTypes => CommandFields.Keys;

    static void ValidateCommandFields(StudioCommand command)
    {
        if (command.Type is null || !CommandFields.TryGetValue(command.Type, out var allowed))
            throw new StudioException("unsupported_command", $"Editing command '{command.Type}' is not supported."); // same refusal as ApplyOne's default
        foreach (var property in typeof(StudioCommand).GetProperties())
            if (property.Name is not ("Type" or "ApplyToLinked") && property.GetValue(command) != null && !allowed.Contains(property.Name, StringComparer.Ordinal))
                throw new StudioException("unsupported_parameter", $"Parameter {property.Name} does not apply to {command.Type}; it would otherwise be ignored.");
    }

    static void ApplyOne(StudioProject p, StudioCommand c)
    {
        switch (c.Type)
        {
            case "deleteAsset":
            {
                var asset = Asset(p, c.AssetId);
                var uses = p.Clips.Count(clip => clip.AssetId == asset.Id);
                if (uses > 0 && c.RemoveTimelineClips != true)
                    throw new StudioException("asset_in_use", $"This asset is used by {uses} timeline clip(s). Confirm removal of those clips before deleting the asset from the library.");
                // This is a library edit, never filesystem deletion. Remove only actual uses,
                // including linked A/V uses of this asset, without deleting unrelated linked media.
                p.Clips.RemoveAll(clip => clip.AssetId == asset.Id);
                p.Assets.Remove(asset);
                break;
            }
            case "addTrack":
                p.Tracks.Add(new() { Id = c.TrackId ?? StudioJson.NewId(), Name = c.Name ?? "Track", Kind = c.Kind ?? "video" }); break;
            case "updateTrack":
            {
                var t = Track(p, c.TrackId);
                if (c.Name != null) t.Name = c.Name;
                if (c.Volume != null) t.Volume = c.Volume.Value;
                if (c.Muted != null) t.Muted = c.Muted.Value;
                if (c.Visible != null) t.Visible = c.Visible.Value;
                break;
            }
            case "reorderTrack":
            {
                var t = Track(p, c.TrackId); var index = c.Index ?? throw new StudioException("missing_parameter", "Track index is required.");
                if (index < 0 || index >= p.Tracks.Count) throw new StudioException("invalid_index", "Track index is outside the project.");
                p.Tracks.Remove(t); p.Tracks.Insert(index, t); break;
            }
            case "removeTrack":
                p.Tracks.Remove(Track(p, c.TrackId)); p.Clips.RemoveAll(x => x.TrackId == c.TrackId); break;
            case "insertClip":
            {
                var a = Asset(p, c.AssetId); Track(p, c.TrackId);
                p.Clips.Add(new()
                {
                    Id = c.ClipId ?? StudioJson.NewId(), AssetId = a.Id, TrackId = c.TrackId!, StartFrame = c.StartFrame ?? 0,
                    SourceStartFrame = c.SourceStartFrame ?? 0,
                    DurationFrames = c.DurationFrames ?? (a.Kind == "image" ? p.FrameRate.Frames(4) : Math.Max(1, (long)Math.Floor(a.DurationSeconds * p.FrameRate.Value))),
                    Volume = c.Volume ?? 1, FadeInFrames = c.FadeInFrames ?? 0, FadeOutFrames = c.FadeOutFrames ?? 0, AudioFadeOutFrames = c.AudioFadeOutFrames ?? 0,
                    LinkGroup = c.LinkGroup, ShotId = c.ShotId, Fit = c.Fit ?? "contain"
                }); break;
            }
            case "moveClip":
            {
                var clip = Clip(p, c.ClipId); var delta = checked(Required(c.StartFrame, "startFrame") - clip.StartFrame);
                foreach (var member in Linked(p, clip, c.ApplyToLinked)) member.StartFrame = checked(member.StartFrame + delta);
                if (c.TrackId != null) { Track(p, c.TrackId); clip.TrackId = c.TrackId; }
                break;
            }
            case "trimClip":
            {
                var clip = Clip(p, c.ClipId); var sourceDelta = checked((c.SourceStartFrame ?? clip.SourceStartFrame) - clip.SourceStartFrame);
                var startDelta = checked((c.StartFrame ?? clip.StartFrame) - clip.StartFrame);
                var durationDelta = checked((c.DurationFrames ?? clip.DurationFrames) - clip.DurationFrames);
                foreach (var member in Linked(p, clip, c.ApplyToLinked))
                {
                    member.SourceStartFrame = checked(member.SourceStartFrame + sourceDelta);
                    member.StartFrame = checked(member.StartFrame + startDelta);
                    member.DurationFrames = checked(member.DurationFrames + durationDelta);
                    member.FadeInFrames = Math.Min(member.FadeInFrames, member.DurationFrames);
                    member.FadeOutFrames = Math.Min(member.FadeOutFrames, Math.Max(0, member.DurationFrames - member.FadeInFrames));
                    member.AudioFadeOutFrames = Math.Min(member.AudioFadeOutFrames, Math.Max(0, member.DurationFrames - member.FadeInFrames));
                }
                break;
            }
            case "updateClip":
            {
                var clip = Clip(p, c.ClipId);
                if (c.Volume != null) clip.Volume = c.Volume.Value;
                if (c.FadeInFrames != null) clip.FadeInFrames = c.FadeInFrames.Value;
                if (c.FadeOutFrames != null) clip.FadeOutFrames = c.FadeOutFrames.Value;
                if (c.AudioFadeOutFrames != null) clip.AudioFadeOutFrames = c.AudioFadeOutFrames.Value;
                if (c.Fit != null) clip.Fit = c.Fit;
                if (c.LinkGroup != null) clip.LinkGroup = c.LinkGroup.Length == 0 ? null : c.LinkGroup;
                if (c.ShotId != null) clip.ShotId = c.ShotId;
                break;
            }
            case "splitClip":
            {
                var clip = Clip(p, c.ClipId); var at = Required(c.AtFrame, "atFrame");
                var group = clip.LinkGroup == null ? null : StudioJson.NewId();
                foreach (var member in Linked(p, clip, c.ApplyToLinked))
                {
                    var offset = checked(at - member.StartFrame);
                    if (offset <= 0 || offset >= member.DurationFrames) throw new StudioException("invalid_split", "Split must lie inside every selected linked clip.");
                    if (offset < member.FadeInFrames || offset > member.DurationFrames - Math.Max(member.FadeOutFrames, member.AudioFadeOutFrames))
                        throw new StudioException("split_inside_fade", "Split outside the active fade region, or clear that fade first; splitting inside a fade would change its visible/audio shape.");
                    var right = StudioJson.Clone(member); right.Id = StudioJson.NewId(); right.StartFrame = at;
                    right.SourceStartFrame = checked(right.SourceStartFrame + offset); right.DurationFrames -= offset;
                    right.FadeInFrames = 0; right.FadeOutFrames = Math.Min(right.FadeOutFrames, right.DurationFrames); right.AudioFadeOutFrames = Math.Min(right.AudioFadeOutFrames, right.DurationFrames); right.LinkGroup = group;
                    member.DurationFrames = offset; member.FadeOutFrames = 0; member.AudioFadeOutFrames = 0; member.FadeInFrames = Math.Min(member.FadeInFrames, offset);
                    p.Clips.Add(right);
                }
                break;
            }
            case "dissolve":
            {
                var incoming = Clip(p, c.ClipId); var outgoing = Clip(p, c.OtherClipId);
                var outgoingEnd = checked(outgoing.StartFrame + outgoing.DurationFrames);
                var incomingEnd = checked(incoming.StartFrame + incoming.DurationFrames);
                if (incoming.Id == outgoing.Id || incoming.TrackId != outgoing.TrackId || Track(p, incoming.TrackId).Kind != "video" ||
                    outgoing.StartFrame >= incoming.StartFrame || incoming.StartFrame >= outgoingEnd || outgoingEnd >= incomingEnd)
                    throw new StudioException("invalid_dissolve", "Dissolve requires two partially overlapping clips on the same video track, with the incoming clip starting and ending later. Position the clips first.");
                var overlap = outgoingEnd - incoming.StartFrame;
                if (p.Clips.Any(x => x.TrackId == incoming.TrackId && x.Id != incoming.Id && x.Id != outgoing.Id && x.StartFrame < outgoingEnd && x.StartFrame + x.DurationFrames > incoming.StartFrame))
                    throw new StudioException("invalid_dissolve", "A dissolve cannot include a third overlapping clip on the same track.");
                if (outgoing.FadeOutFrames != 0 || outgoing.FadeInFrames > incoming.StartFrame - outgoing.StartFrame ||
                    outgoing.AudioFadeOutFrames != 0 && outgoing.AudioFadeOutFrames != overlap || incoming.FadeInFrames != 0 && incoming.FadeInFrames != overlap ||
                    Math.Max(incoming.FadeOutFrames, incoming.AudioFadeOutFrames) > incoming.DurationFrames - overlap)
                    throw new StudioException("conflicting_fade", "Clear conflicting clip fades before applying this dissolve.");
                // Keep the outgoing picture opaque. Only the incoming alpha ramps, so the two
                // pictures sum to one rather than dipping to black. Audio uses complementary
                // linear gains; linked audio companions receive the same source-time envelope.
                incoming.FadeInFrames = overlap; outgoing.AudioFadeOutFrames = overlap;
                foreach (var member in Linked(p, outgoing, true).Where(x => x.Id != outgoing.Id && Track(p, x.TrackId).Kind == "audio"))
                {
                    if (member.StartFrame + member.DurationFrames != outgoingEnd || member.StartFrame > incoming.StartFrame || member.FadeInFrames > member.DurationFrames - overlap ||
                        member.FadeOutFrames != 0 && member.FadeOutFrames != overlap || member.AudioFadeOutFrames != 0 && member.AudioFadeOutFrames != overlap)
                        throw new StudioException("conflicting_link", "Linked outgoing audio must cover the overlap, end with its picture, and have no conflicting fade.");
                    member.AudioFadeOutFrames = overlap;
                }
                foreach (var member in Linked(p, incoming, true).Where(x => x.Id != incoming.Id && Track(p, x.TrackId).Kind == "audio"))
                {
                    if (member.StartFrame != incoming.StartFrame || member.DurationFrames < overlap || member.FadeInFrames != 0 && member.FadeInFrames != overlap ||
                        Math.Max(member.FadeOutFrames, member.AudioFadeOutFrames) > member.DurationFrames - overlap)
                        throw new StudioException("conflicting_link", "Linked incoming audio must start with its picture and have no conflicting fade.");
                    member.FadeInFrames = overlap;
                }
                break;
            }
            case "duplicateClip":
            {
                var clip = Clip(p, c.ClipId); var delta = checked((c.StartFrame ?? checked(clip.StartFrame + clip.DurationFrames)) - clip.StartFrame);
                var group = clip.LinkGroup == null ? null : StudioJson.NewId();
                foreach (var member in Linked(p, clip, c.ApplyToLinked))
                {
                    var copy = StudioJson.Clone(member); copy.Id = StudioJson.NewId(); copy.StartFrame = checked(copy.StartFrame + delta); copy.LinkGroup = group; p.Clips.Add(copy);
                }
                break;
            }
            case "removeClip":
            {
                var clip = Clip(p, c.ClipId); var ids = Linked(p, clip, c.ApplyToLinked).Select(x => x.Id).ToHashSet();
                p.Clips.RemoveAll(x => ids.Contains(x.Id)); break;
            }
            case "replaceAsset":
            {
                var clip = Clip(p, c.ClipId); var asset = Asset(p, c.AssetId); clip.AssetId = asset.Id; break;
            }
            case "addCaption":
                p.Captions.Add(new() { Id = c.CaptionId ?? StudioJson.NewId(), Text = c.Text ?? "", StartFrame = Required(c.StartFrame, "startFrame"), DurationFrames = Required(c.DurationFrames, "durationFrames") }); break;
            case "updateCaption":
            {
                var caption = Caption(p, c.CaptionId);
                if (c.Text != null) caption.Text = c.Text;
                if (c.StartFrame != null) caption.StartFrame = c.StartFrame.Value;
                if (c.DurationFrames != null) caption.DurationFrames = c.DurationFrames.Value;
                break;
            }
            case "removeCaption": p.Captions.Remove(Caption(p, c.CaptionId)); break;
            case "settings":
                if (c.ConversationId != null) p.ConversationId = c.ConversationId.Length == 0 ? null : c.ConversationId;
                if (c.Name != null) p.Name = c.Name;
                if (c.Width != null) p.Width = c.Width.Value;
                if (c.Height != null) p.Height = c.Height.Value;
                break;
            default: throw new StudioException("unsupported_command", $"Editing command '{c.Type}' is not supported.");
        }
    }
}
