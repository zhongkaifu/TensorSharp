namespace GlmUnifiedSessionProbe;

internal static class FinalLogitCapture
{
    internal static float[] Copy(float[]? values, int vocabulary)
    {
        if (vocabulary < 1 || values == null || values.Length != vocabulary || !values.All(float.IsFinite))
            throw new InvalidDataException("Final model logits must contain the complete finite vocabulary.");
        return (float[])values.Clone();
    }

    internal static void SelfTest()
    {
        float[] source = [1f, -2f, 0f];
        float[] copy = Copy(source, 3); source[0] = 4f;
        if (copy[0] != 1f || ReferenceEquals(source, copy)) throw new Exception("Logit capture aliases a mutable model buffer.");
        int rejected = 0;
        foreach (var (values, vocabulary) in new (float[]?, int)[] {
            (null, 3), ([], 0), ([1f], 2), ([float.NaN], 1), ([float.PositiveInfinity], 1), ([float.NegativeInfinity], 1) })
            try { Copy(values, vocabulary); } catch (InvalidDataException) { rejected++; }
        if (rejected != 6) throw new Exception("Invalid logit capture was accepted.");
        Console.WriteLine("PASS 7 final-logit copy/isolation/finite/shape checks; no model or native execution.");
    }
}
