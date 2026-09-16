using System.Runtime.InteropServices;
using TensorSharp.GGML;

namespace InferenceWeb.Tests;

public sealed class Glm5NextNativeSnapshotBoundaryTests
{
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate void InjectFault(int stage, int kind);

    [GlmSnapshotBoundaryTheory]
    [InlineData(1)]
    [InlineData(2)]
    public void CaptureAndPartialRestoreExceptionsAreContainedAndResetRecovers(int kind)
        => CheckCaptureAndPartialRestoreExceptions(kind, "CPU");

    [GlmSnapshotBoundaryCudaTheory]
    [InlineData(1)]
    [InlineData(2)]
    public void CudaCaptureAndPartialRestoreExceptionsAreContainedAndResetRecovers(int kind)
        => CheckCaptureAndPartialRestoreExceptions(kind, "CUDA");

    private static void CheckCaptureAndPartialRestoreExceptions(int kind, string backend)
    {
        string directory = Path.Combine(Path.GetTempPath(), "ts-glm-kda-boundary-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(directory);
        IntPtr handle = IntPtr.Zero, library = IntPtr.Zero;
        InjectFault inject = null;
        try
        {
            string path = GlmDsaSyntheticModelBuilder.WriteGlm5NextTpFixture(
                Path.Combine(directory, "model.gguf"), 4, false, numLayers: 2);
            handle = GgmlGlmNative.LoadModel(path, 1, 256, 16, 2, backendName: backend, ctxIsHardLimit: true);
            Assert.NotEqual(IntPtr.Zero, handle);
            library = NativeLibrary.Load(TestGates.MappedNativeGgmlOpsPath());
            inject = Marshal.GetDelegateForFunctionPointer<InjectFault>(
                NativeLibrary.GetExport(library, "TSGgml_GlmTestKdaSnapshotFault"));
            int[] prompt = { 65, 66, 67, 68, 69 };
            var expected = new float[GgmlGlmNative.VocabSize(handle)];
            Assert.True(GgmlGlmNative.Forward(handle, prompt, expected));
            Assert.True(GgmlGlmNative.Forward(handle, new[] { 70 }, expected));
            Assert.True(GgmlGlmNative.ResetChecked(handle));
            var actual = new float[expected.Length];
            Assert.True(GgmlGlmNative.Forward(handle, prompt, actual));

            inject(1, kind);
            Assert.False(GgmlGlmNative.KdaStateCapture(handle));
            Assert.Equal(prompt.Length, GgmlGlmNative.NPast(handle));
            Assert.Equal(-1, GgmlGlmNative.KdaStateRestore(handle));
            Assert.True(GgmlGlmNative.Forward(handle, new[] { 70 }, actual));
            Assert.Equal(expected, actual);

            Assert.True(GgmlGlmNative.KdaStateCapture(handle));
            Assert.True(GgmlGlmNative.Forward(handle, new[] { 71, 72 }, actual));
            inject(2, kind); // after the first conv tensor was copied back
            Assert.Equal(-1, GgmlGlmNative.KdaStateRestore(handle));
            Array.Fill(actual, -123f);
            Assert.False(GgmlGlmNative.Forward(handle, new[] { 73 }, actual));
            Assert.False(GgmlGlmNative.SpecForward(handle, new[] { 73 }, null, actual, false));
            Assert.False(GgmlGlmNative.Rewind(handle, 0));
            Assert.False(GgmlGlmNative.KdaStateCapture(handle));
            Assert.All(actual, value => Assert.Equal(-123f, value));

            int peer = GgmlGlmNative.SlotAlloc(handle);
            Assert.True(peer > 0);
            Assert.True(GgmlGlmNative.SetActiveSlot(handle, peer));
            Assert.True(GgmlGlmNative.Forward(handle, prompt, actual));
            Assert.True(GgmlGlmNative.Forward(handle, new[] { 70 }, actual));
            Assert.Equal(expected, actual);
            Assert.True(GgmlGlmNative.SetActiveSlot(handle, 0));
            Assert.True(GgmlGlmNative.ResetChecked(handle));
            Assert.True(GgmlGlmNative.Forward(handle, prompt, actual));
            Assert.True(GgmlGlmNative.Forward(handle, new[] { 70 }, actual));
            Assert.Equal(expected, actual);
        }
        finally
        {
            inject?.Invoke(0, 0);
            if (handle != IntPtr.Zero) GgmlGlmNative.Free(handle);
            if (library != IntPtr.Zero) NativeLibrary.Free(library);
            Directory.Delete(directory, true);
        }
    }
}
