using System.Diagnostics;
using System.Text.Json;

static class ProbeChecks
{
    public static void ValidateLogits(float[] logits, int vocabulary)
    {
        if (logits.Length != vocabulary || logits.Any(v => !float.IsFinite(v)))
            throw new InvalidDataException("Nonfinite/incomplete actual logits.");
    }

    public static void ForceTeacher(float[] logits, int teacher)
    {
        if ((uint)teacher >= (uint)logits.Length) throw new ArgumentOutOfRangeException(nameof(teacher));
        Array.Fill(logits, float.MinValue); logits[teacher] = float.MaxValue;
    }

    public static bool ExactShape(int[] deltas, int prompt, int chunk) => deltas.Length != 0
        && deltas.Sum() == prompt && deltas.All(x => x > 0 && x <= chunk) && deltas[0] == chunk;

    public static int Run()
    {
        int count = 0;
        void Check(bool value) { if (!value) throw new Exception("Self-test failed: " + count); count++; }
        void Refuses(Action action) { bool refused = false; try { action(); } catch (ArgumentException) { refused = true; }
            catch (InvalidDataException) { refused = true; } Check(refused); }
        ValidateLogits([1, -2, 3], 3); count++;
        Refuses(() => ValidateLogits([1, float.NaN], 2));
        Refuses(() => ValidateLogits([1, float.PositiveInfinity], 2));
        Refuses(() => ValidateLogits([1], 2));
        float[] actual = [1, 2, 3], captured = (float[])actual.Clone(); ForceTeacher(actual, 1);
        Check(captured.SequenceEqual([1, 2, 3])); Check(actual[1] == float.MaxValue);
        Check(actual[0] == float.MinValue && actual[2] == float.MinValue);
        Refuses(() => ForceTeacher(actual, -1)); Refuses(() => ForceTeacher(actual, 3));
        Check(ExactShape([317], 317, 317)); Check(ExactShape([32,32,32,32,32,32,32,32,32,29], 317, 32));
        Check(!ExactShape([128,128,61], 317, 317)); Check(!ExactShape([], 317, 32));
        Check(!ExactShape([32, 29], 317, 32)); Check(!ExactShape([32, 285], 317, 32));
        Check(!ExactShape([32, 0, 285], 317, 32));
        var native = Process.GetCurrentProcess().Modules.Cast<ProcessModule>().Where(m => m.ModuleName.Contains("GgmlOps", StringComparison.OrdinalIgnoreCase)).ToArray();
        Check(native.Length == 0);
        Console.WriteLine(JsonSerializer.Serialize(new { Schema = "qwen-image-logit-selftest.v1", Passed = count, NativeModules = native.Length }));
        return 0;
    }
}
