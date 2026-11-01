// Model-free tests link the exact schedule implementation; no TensorSharp model,
// native backend, network, or third-party test framework is loaded.
using System.Text.Json;

var json = new JsonSerializerOptions { PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower };
int checks = 0;
void Check(bool condition, string name)
{
    if (!condition) throw new Exception(name);
    checks++;
}
void Reject(Action action, string name)
{
    try { action(); }
    catch (ArgumentException) { checks++; return; }
    throw new Exception("Expected rejection: " + name);
}
ProbeCase Recorded(string name, int length) => new()
{
    Name = name, PromptTokens = Enumerable.Range(0, length).ToArray(), GeneratedTokens = [4, 5, 6],
};

var shortCase = Recorded("short", 2);
var longCase = Recorded("long", 128);
var external = ProbeSchedule.Validate([shortCase, longCase]);
foreach (string[] order in new[] { new[] { "short", "short" }, new[] { "long", "long" },
                                  new[] { "short", "long", "short" }, new[] { "long", "short" } })
{
    var schedule = order.Select((name, i) => new ProbeCase { Name = "case" + i, ReferenceCase = name }).ToArray();
    ProbeSchedule.Validate(schedule);
    ProbeSchedule.ValidateReferences(schedule, external);
    foreach (var item in schedule)
    {
        var expected = external[item.ReferenceCase!];
        ProbeSchedule.UseReferenceHistory(item, expected);
        Check(item.PromptTokens!.SequenceEqual(expected.PromptTokens!), "identical prompt history");
        Check(item.GeneratedTokens!.SequenceEqual(expected.GeneratedTokens!), "teacher forced generated history");
        Check(!ReferenceEquals(item.PromptTokens, expected.PromptTokens), "history must be independently owned");
    }
}
ProbeSchedule.ValidateReferences([new() { Name = "first", PromptTokens = [1] },
    new() { Name = "again", ReferenceCase = "first" }], null);
checks++;
Reject(() => ProbeSchedule.ValidateReferences([new() { Name = "first", ReferenceCase = "later" },
    new() { Name = "later", PromptTokens = [1] }], null), "forward reference");
Reject(() => ProbeSchedule.ValidateReferences([new() { Name = "first", ReferenceCase = "first" }], null), "self reference");
Reject(() => ProbeSchedule.ValidateReferences([new() { Name = "first", ReferenceCase = "absent" }], external), "missing external");
Reject(() => ProbeSchedule.Validate([new() { Name = "same" }, new() { Name = "SAME" }]), "duplicate filename");
foreach (string name in new[] { "../escape", "x/y", "x\\y", "x\n", ".", "..", "CON", "a." })
    Reject(() => ProbeSchedule.Validate([new() { Name = name }]), "unsafe filename");
Reject(() => ProbeSchedule.Validate([]), "empty schedule");
Reject(() => ProbeSchedule.Validate([new() { Name = "negative", PromptTokens = [-1] }]), "negative token");
Reject(() => ProbeSchedule.UseReferenceHistory(new() { Name = "bad", PromptTokens = [99] }, shortCase), "prompt mismatch");
Reject(() => ProbeSchedule.UseReferenceHistory(new() { Name = "bad", GeneratedTokens = [99] }, shortCase), "decode mismatch");
Reject(() => ProbeSchedule.UseReferenceHistory(new() { Name = "bad", TargetTokens = 128 }, shortCase), "length mismatch");
Reject(() => ProbeSchedule.UseReferenceHistory(new() { Name = "bad" }, new() { Name = "empty", PromptTokens = [1] }), "unfinished reference");
var parsed = JsonSerializer.Deserialize<ProbeCase[]>("[{\"name\":\"alias\",\"reference_case\":\"short\"}]", json)!;
Check(parsed[0].ReferenceCase == "short", "JSON snake_case alias");
ProbeSchedule.UseReferenceHistory(parsed[0], shortCase);
var roundtrip = JsonSerializer.Deserialize<ProbeCase[]>(JsonSerializer.Serialize(parsed, json), json)!;
ProbeSchedule.ValidateReferences(roundtrip, external);
Check(roundtrip[0].PromptTokens!.SequenceEqual(shortCase.PromptTokens!), "recorded JSON round trip");
Check(ProbeSchedule.TokenHash([0, 1, 2]) == "ad5dc1478de06a4c2728ea528bd9361a4b945e92a414bf4d180cedaaeaa5f4cc", "portable int32-le SHA256");
Check(ProbeSchedule.TokenHash([1, 2]) != ProbeSchedule.TokenHash([2, 1]), "hash preserves order");
Console.WriteLine($"PASS: {checks} model-free forced-logit schedule assertions");
