using TensorSharp.Models.Ltx25;
using TensorSharp.Models.SharedDiffusion;

namespace InferenceWeb.Tests;

/// <summary>
/// One LTX 2.5 sampling pass as the diffusion runtime runs it, with the prediction supplied by the test: each step's denoiser
/// input from <c>Ltx25Family.StepInput</c>, then the recipe's conditioned sampler on each stream (the video stream with its
/// guide mask and clean latents, ancestral noise where <paramref name="noise"/> is given and the step does not end at 0).
/// </summary>
internal static class Ltx25SamplerLoop
{
    public sealed record Result(float[] Video, float[] Audio, int Steps);

    public static Result Run(Func<Ltx25DenoiserInput, Ltx25DenoiserOutput> predict, Ltx25DenoiserInput initial, float[] sigmas,
        Func<int, (float[] Video, float[] Audio)>? noise = null, Action<int, int>? onStep = null)
    {
        IConditionedSampler sampler = noise == null ? new EulerDenoisedSampler() : new EulerAncestralSampler();
        var video = (float[])initial.Video.Clone();
        var audio = (float[])initial.Audio.Clone();
        int steps = sigmas.Length - 1;
        for (int step = 0; step < steps; ++step)
        {
            float sigma = sigmas[step], next = sigmas[step + 1];
            Ltx25DenoiserInput input = Ltx25Family.StepInput(initial, video, audio, sigma, initial.VideoContext, initial.AudioContext,
                initial.VideoContextTokens, initial.UseReferenceLora);
            Ltx25DenoiserOutput prediction = predict(input);
            (float[]? Video, float[]? Audio) drawn = noise != null && next != 0 ? noise(step) : (null, null);
            sampler.Step(video, prediction.VideoVelocity, sigma, next,
                new SamplerStreamInputs { Noise = drawn.Video, DenoiseMask = initial.VideoDenoiseMask, Clean = initial.CleanVideo });
            sampler.Step(audio, prediction.AudioVelocity, sigma, next, new SamplerStreamInputs { Noise = drawn.Audio });
            onStep?.Invoke(step + 1, steps);
        }
        return new(video, audio, steps);
    }
}
