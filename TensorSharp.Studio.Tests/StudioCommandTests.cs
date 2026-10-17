using System.Text.Json;
using TensorSharp.Studio;

namespace TensorSharp.Studio.Tests;

public class StudioCommandTests
{
    static StudioProject Project()
    {
        var p = new StudioProject { Width = 320, Height = 180 };
        p.Tracks.Add(new() { Name = "Video" }); p.Tracks.Add(new() { Name = "Audio", Kind = "audio" });
        p.Assets.Add(new() { FileName = "fixture.mp4", DurationSeconds = 10, HasAudio = true, Sha256 = new('0', 64), Bytes = 20 });
        return p;
    }
    static StudioCommand Insert(StudioProject p, string? id = null, string? link = null, int track = 0) => new()
    { Type = "insertClip", ClipId = id, AssetId = p.Assets[0].Id, TrackId = p.Tracks[track].Id, DurationFrames = 120, LinkGroup = link };

    [Fact] public void FractionalFrameRatesUseSpecifiedRoundingAndRoundTrip()
    {
        var rate = new StudioFrameRate(30000, 1001); rate.Validate();
        Assert.Equal(30000, rate.Frames(1001)); Assert.Equal(1001, rate.Seconds(30000), 8);
        Assert.Equal(1, new StudioFrameRate(24, 1).Frames(.5 / 24));
        Assert.Throws<StudioException>(() => new StudioFrameRate(1, 0).Validate());
    }
    [Fact] public void LinkedMoveTrimSplitAndDeletePreserveSourceAssets()
    {
        var p = Project(); var group = StudioJson.NewId();
        StudioCommands.Apply(p, [Insert(p, link: group), Insert(p, link: group, track: 1)], new());
        var selected = p.Clips[0].Id;
        StudioCommands.Apply(p, [new() { Type = "moveClip", ClipId = selected, StartFrame = 24 },
            new() { Type = "trimClip", ClipId = selected, SourceStartFrame = 24, DurationFrames = 96 },
            new() { Type = "splitClip", ClipId = selected, AtFrame = 72 }], new());
        Assert.Equal(4, p.Clips.Count); Assert.All(p.Clips, c => Assert.Equal(48, c.DurationFrames));
        Assert.Equal(2, p.Clips.Select(c => c.LinkGroup).Distinct().Count());
        Assert.Equal(2, p.Clips.Count(c => c.SourceStartFrame == 72));
        StudioCommands.Apply(p, [new() { Type = "removeClip", ClipId = selected }], new());
        Assert.Equal(2, p.Clips.Count); Assert.Single(p.Assets);
    }
    [Fact] public void RemoveTrackDeletesOnlyItsOwnClipsAndKeepsLinkedPartnersCaptionsAndAssets()
    {
        var p = Project(); var group = StudioJson.NewId();
        StudioCommands.Apply(p, [Insert(p, link: group), Insert(p, link: group, track: 1), Insert(p, track: 1),
            new() { Type = "addCaption", StartFrame = 0, DurationFrames = 24, Text = "Hello" }], new());
        var video = p.Tracks[0].Id; var audio = p.Tracks[1].Id; var picture = p.Clips[0].Id;
        StudioCommands.Apply(p, [new() { Type = "removeTrack", TrackId = audio }], new());
        Assert.Equal(video, Assert.Single(p.Tracks).Id);
        var kept = Assert.Single(p.Clips); Assert.Equal(picture, kept.Id); Assert.Equal(group, kept.LinkGroup); // removeTrack never reaches across tracks
        Assert.Single(p.Captions); Assert.Single(p.Assets);
        Assert.Equal("track_not_found", Assert.Throws<StudioException>(() => StudioCommands.Apply(p, [new() { Type = "removeTrack", TrackId = audio }], new())).Code);
        Assert.Equal("unsupported_parameter", Assert.Throws<StudioException>(() => StudioCommands.Apply(p, [new() { Type = "removeTrack", TrackId = video, Name = "Video" }], new())).Code);
        StudioCommands.Apply(p, [new() { Type = "removeTrack", TrackId = video }], new());
        Assert.Empty(p.Tracks); Assert.Empty(p.Clips); Assert.Single(p.Captions); Assert.Single(p.Assets);
    }
    [Fact] public void EveryAdvertisedCommandTypeReachesItsInterpreterCase()
    {
        Assert.Contains("removeTrack", StudioCommands.CommandTypes);
        Assert.Equal("unsupported_command", Assert.Throws<StudioException>(() => StudioCommands.Apply(Project(), [new() { Type = "RemoveTrack", TrackId = StudioJson.NewId() }], new())).Code); // type names are exact
        foreach (var type in StudioCommands.CommandTypes)
            Assert.False(Record.Exception(() => StudioCommands.Apply(Project(), [new() { Type = type }], new())) is StudioException { Code: "unsupported_command" }, type);
    }
    [Fact] public void LinkedEditingKeepsRelativeOffsetsAndAtomicServiceValidationPreventsInvalidRanges()
    {
        var p = Project(); var group = StudioJson.NewId(); var picture = Insert(p, link: group); var sound = Insert(p, link: group, track: 1);
        picture.StartFrame = 24; picture.SourceStartFrame = 12;
        sound.StartFrame = 30; sound.SourceStartFrame = 48;
        StudioCommands.Apply(p, [picture, sound], new()); var selected = p.Clips[0].Id;
        StudioCommands.Apply(p, [new() { Type = "moveClip", ClipId = selected, StartFrame = 48 }, new() { Type = "trimClip", ClipId = selected, SourceStartFrame = 24, DurationFrames = 96 }, new() { Type = "splitClip", ClipId = selected, AtFrame = 96 }], new());
        var leftVideo = p.Clips.Single(c => c.Id == selected); var leftAudio = p.Clips.Single(c => c.TrackId == p.Tracks[1].Id && c.StartFrame == 54);
        Assert.Equal(6, leftAudio.StartFrame - leftVideo.StartFrame); Assert.Equal(36, leftAudio.SourceStartFrame - leftVideo.SourceStartFrame);
        var right = p.Clips.Where(c => c.StartFrame == 96).ToArray(); Assert.Equal(2, right.Length); Assert.Single(right.Select(c => c.LinkGroup).Distinct());
        Assert.Equal(72, right.Single(c => c.TrackId == p.Tracks[0].Id).SourceStartFrame);
        Assert.Equal(102, right.Single(c => c.TrackId == p.Tracks[1].Id).SourceStartFrame);
        Assert.Equal(48, right.Single(c => c.TrackId == p.Tracks[0].Id).DurationFrames);
        Assert.Equal(54, right.Single(c => c.TrackId == p.Tracks[1].Id).DurationFrames);
    }
    [Fact] public void DissolvePreservesTimingAndAppliesComplementaryFadesToLinkedAudio()
    {
        var p = Project(); var firstGroup = StudioJson.NewId(); var secondGroup = StudioJson.NewId();
        var first = Insert(p, link: firstGroup); first.DurationFrames = 48;
        var firstAudio = Insert(p, link: firstGroup, track: 1); firstAudio.DurationFrames = 48;
        var second = Insert(p, link: secondGroup); second.StartFrame = 24; second.DurationFrames = 48;
        var secondAudio = Insert(p, link: secondGroup, track: 1); secondAudio.StartFrame = 24; secondAudio.DurationFrames = 48;
        StudioCommands.Apply(p, [first, firstAudio, second, secondAudio], new());
        var outgoing = p.Clips[0]; var incoming = p.Clips[2];
        StudioCommands.Apply(p, [new() { Type = "dissolve", ClipId = incoming.Id, OtherClipId = outgoing.Id }], new());
        Assert.Equal(24, incoming.StartFrame); Assert.Equal(48, incoming.DurationFrames); Assert.Equal(72, p.DurationFrames);
        Assert.Equal(0, outgoing.FadeOutFrames); Assert.Equal(24, outgoing.AudioFadeOutFrames);
        Assert.All(p.Clips.Where(c => c.LinkGroup == firstGroup), c => Assert.Equal(24, c.AudioFadeOutFrames));
        Assert.All(p.Clips.Where(c => c.LinkGroup == secondGroup), c => Assert.Equal(24, c.FadeInFrames));
    }
    [Fact] public void DissolveRefusesUnrelatedOrTripleOverlappingClipsAndSplittingInsideFades()
    {
        var p = Project(); StudioCommands.Apply(p, [Insert(p), Insert(p)], new());
        Assert.Equal("invalid_dissolve", Assert.Throws<StudioException>(() => StudioCommands.Apply(p, [new() { Type = "dissolve", ClipId = p.Clips[1].Id, OtherClipId = p.Clips[0].Id }], new())).Code);
        p.Clips[1].StartFrame = 60;
        StudioCommands.Apply(p, [new() { Type = "insertClip", TrackId = p.Tracks[0].Id, AssetId = p.Assets[0].Id, StartFrame = 90, DurationFrames = 12 }], new());
        Assert.Equal("invalid_dissolve", Assert.Throws<StudioException>(() => StudioCommands.Apply(p, [new() { Type = "dissolve", ClipId = p.Clips[1].Id, OtherClipId = p.Clips[0].Id }], new())).Code);
        p.Clips.RemoveAt(2); StudioCommands.Apply(p, [new() { Type = "dissolve", ClipId = p.Clips[1].Id, OtherClipId = p.Clips[0].Id }], new());
        Assert.Equal("split_inside_fade", Assert.Throws<StudioException>(() => StudioCommands.Apply(p, [new() { Type = "splitClip", ClipId = p.Clips[1].Id, AtFrame = 90 }], new())).Code);
    }
    [Theory]
    [InlineData("invalid_trim", -1, 10)]
    [InlineData("invalid_trim", 239, 24)]
    [InlineData("invalid_range", 0, 0)]
    public void SourceBoundsAndDurationsAreValidated(string code, long source, long duration)
    {
        var p = Project(); var insert = Insert(p); insert.SourceStartFrame = source; insert.DurationFrames = duration;
        Assert.Equal(code, Assert.Throws<StudioException>(() => StudioCommands.Apply(p, [insert], new())).Code);
    }
    [Fact] public void RejectsUnsupportedCommandsCrossProjectAssetsAndInvalidEffects()
    {
        var p = Project();
        Assert.Equal("unsupported_command", Assert.Throws<StudioException>(() => StudioCommands.Apply(p, [new() { Type = "runShell", Text = "rm -rf /" }], new())).Code);
        var insert = Insert(p); insert.AssetId = StudioJson.NewId();
        Assert.Equal("asset_not_found", Assert.Throws<StudioException>(() => StudioCommands.Apply(p, [insert], new())).Code);
        StudioCommands.Apply(p, [Insert(p)], new());
        Assert.Equal("invalid_fade", Assert.Throws<StudioException>(() => StudioCommands.Apply(p, [new() { Type = "updateClip", ClipId = p.Clips[0].Id, FadeInFrames = 100, FadeOutFrames = 100 }], new())).Code);
    }
    [Fact] public void OverlayTracksAndCaptionsSerializeWithoutMediaBytes()
    {
        var p = Project(); StudioCommands.Apply(p, [Insert(p), new() { Type = "addCaption", StartFrame = 4, DurationFrames = 24, Text = "Hello" }], new());
        var json = JsonSerializer.Serialize(p, StudioJson.Options); var restored = JsonSerializer.Deserialize<StudioProject>(json, StudioJson.Options)!;
        StudioCommands.Validate(restored, new()); Assert.Equal(120, restored.DurationFrames);
        Assert.Contains("\"frameRate\"", json); Assert.DoesNotContain("base64", json); Assert.Equal("Hello", restored.Captions.Single().Text);
    }
    [Fact] public void RejectsFutureSchemasUnsafePathsAndDuplicateIds()
    {
        var p = Project(); p.SchemaVersion = 2;
        Assert.Equal("unsupported_schema", Assert.Throws<StudioException>(() => StudioCommands.Validate(p, new())).Code);
        p.SchemaVersion = 1; p.Assets[0].FileName = "../outside.mp4";
        Assert.Equal("invalid_path", Assert.Throws<StudioException>(() => StudioCommands.Validate(p, new())).Code);
        p.Assets[0].FileName = "fixture.mp4"; p.Tracks.Add(StudioJson.Clone(p.Tracks[0]));
        Assert.Equal("duplicate_id", Assert.Throws<StudioException>(() => StudioCommands.Validate(p, new())).Code);
    }
}
