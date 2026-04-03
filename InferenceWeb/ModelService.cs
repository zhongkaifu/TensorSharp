using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;
using InferenceEngine;
using TensorSharp;
using TensorSharp.Cpu;

namespace InferenceWeb
{
    public class ModelService : IDisposable
    {
        private ModelBase _model;
        private readonly SemaphoreSlim _inferenceLock = new(1, 1);
        private string _loadedModelPath;
        private string _loadedMmProjPath;
        private BackendType _backend;

        public bool IsLoaded => _model != null;
        public string LoadedModelName => _loadedModelPath != null ? Path.GetFileName(_loadedModelPath) : null;
        public string Architecture => _model?.Config?.Architecture;

        public void LoadModel(string modelPath, string mmProjPath, string backendStr)
        {
            _inferenceLock.Wait();
            try
            {
                _model?.Dispose();
                _model = null;
                _loadedModelPath = null;
                _loadedMmProjPath = null;

                _backend = backendStr switch
                {
                    "ggml_metal" => BackendType.GgmlMetal,
                    "ggml_cpu" => BackendType.GgmlCpu,
                    "cpu" => BackendType.Cpu,
                    _ => BackendType.GgmlCpu
                };

                _model = ModelBase.Create(modelPath, _backend);
                _loadedModelPath = modelPath;

                if (mmProjPath == null)
                    mmProjPath = AutoDetectMmProj(modelPath);

                if (mmProjPath != null && File.Exists(mmProjPath))
                {
                    LoadEncoders(mmProjPath);
                    _loadedMmProjPath = mmProjPath;
                }
            }
            finally
            {
                _inferenceLock.Release();
            }
        }

        private string AutoDetectMmProj(string modelPath)
        {
            string dir = Path.GetDirectoryName(modelPath);
            if (dir == null) return null;

            string arch = _model?.Config?.Architecture;
            string[] candidates = arch switch
            {
                "gemma4" => new[] { "gemma-4-mmproj-F16.gguf" },
                "gemma3" => new[] { "mmproj-gemma3-4b-f16.gguf" },
                "qwen35" or "qwen35moe" or "qwen3next" => new[] { "Qwen3.5-mmproj-F16.gguf" },
                _ => Array.Empty<string>()
            };

            foreach (var c in candidates)
            {
                string full = Path.Combine(dir, c);
                if (File.Exists(full)) return full;
            }

            foreach (var f in Directory.GetFiles(dir, "*mmproj*"))
                return f;

            return null;
        }

        private void LoadEncoders(string mmProjPath)
        {
            switch (_model)
            {
                case Gemma4Model g4:
                    g4.LoadVisionEncoder(mmProjPath);
                    g4.LoadAudioEncoder(mmProjPath);
                    break;
                case Gemma3Model g3:
                    g3.LoadVisionEncoder(mmProjPath);
                    break;
                case Qwen35Model q35:
                    q35.LoadVisionEncoder(mmProjPath);
                    break;
            }
        }

        public async IAsyncEnumerable<string> ChatStreamAsync(
            List<ChatMessage> history,
            int maxTokens,
            [EnumeratorCancellation] CancellationToken cancellationToken)
        {
            await _inferenceLock.WaitAsync(cancellationToken);
            try
            {
                string arch = _model.Config.Architecture;
                string rendered = ChatTemplate.RenderFromGgufTemplate(
                    _model.Config.ChatTemplate, history, addGenerationPrompt: true,
                    architecture: arch);

                var inputTokens = _model.Tokenizer.Encode(rendered, addSpecial: true);

                var lastMsg = history.LastOrDefault(m => m.Role == "user");
                if (lastMsg != null)
                    inputTokens = ProcessMultimodal(lastMsg, inputTokens, arch);

                _model.ResetKVCache();
                float[] logits = _model.Forward(inputTokens.ToArray());
                var generatedTokens = new List<int>();

                for (int step = 0; step < maxTokens; step++)
                {
                    if (cancellationToken.IsCancellationRequested)
                        break;

                    int nextToken = _model.SampleGreedy(logits);
                    if (_model.Tokenizer.IsEos(nextToken))
                        break;

                    generatedTokens.Add(nextToken);
                    string piece = _model.Tokenizer.Decode(generatedTokens);
                    if (generatedTokens.Count > 1)
                    {
                        string prev = _model.Tokenizer.Decode(generatedTokens.GetRange(0, generatedTokens.Count - 1));
                        piece = piece.Substring(prev.Length);
                    }

                    yield return piece;

                    logits = _model.Forward(new[] { nextToken });
                }
            }
            finally
            {
                _inferenceLock.Release();
            }
        }

        private List<int> ProcessMultimodal(ChatMessage msg, List<int> inputTokens, string arch)
        {
            if (msg.ImagePaths != null && msg.ImagePaths.Count > 0)
                inputTokens = ProcessImages(msg, inputTokens, arch);

            if (msg.AudioPaths != null && msg.AudioPaths.Count > 0)
                inputTokens = ProcessAudio(msg, inputTokens, arch);

            return inputTokens;
        }

        private List<int> ProcessImages(ChatMessage msg, List<int> inputTokens, string arch)
        {
            if (arch == "gemma4" && _model is Gemma4Model g4 && g4.VisionEncoder != null)
            {
                int imageStartId = _model.Tokenizer.LookupToken("<|image>");
                int imageEndId = _model.Tokenizer.LookupToken("<image|>");
                if (imageStartId < 0) imageStartId = 255999;
                if (imageEndId < 0) imageEndId = 256000;

                var proc = new Gemma4ImageProcessor();
                int searchFrom = 0;
                foreach (var imgPath in msg.ImagePaths)
                {
                    var (pixels, imgW, imgH) = proc.ProcessImage(imgPath);
                    var emb = g4.VisionEncoder.Encode(pixels, imgW, imgH);
                    int numTokens = (int)emb.Sizes[0];

                    int pos = -1;
                    for (int i = searchFrom; i < inputTokens.Count; i++)
                    {
                        if (inputTokens[i] == imageStartId) { pos = i; break; }
                    }

                    if (pos >= 0)
                    {
                        var expanded = new List<int>();
                        for (int i = 0; i < pos; i++) expanded.Add(inputTokens[i]);
                        expanded.Add(imageStartId);
                        for (int i = 0; i < numTokens; i++) expanded.Add(0);
                        expanded.Add(imageEndId);
                        for (int i = pos + 1; i < inputTokens.Count; i++) expanded.Add(inputTokens[i]);
                        inputTokens = expanded;

                        g4.SetVisionEmbeddings(emb, pos + 1);
                        searchFrom = pos + 1 + numTokens + 1;
                    }
                    else
                    {
                        emb.Dispose();
                    }
                }
            }
            else if (arch == "gemma3" && _model is Gemma3Model g3 && g3.VisionEncoder != null)
            {
                var proc = new Gemma3ImageProcessor();
                int startId = _model.Tokenizer.LookupToken("<start_of_image>");
                if (startId < 0) startId = Gemma3ImageProcessor.StartOfImageToken;
                int endId = Gemma3ImageProcessor.EndOfImageToken;
                int nlnlId = Gemma3ImageProcessor.NewlineNewlineToken;
                int padId = Gemma3ImageProcessor.PadToken;

                inputTokens = ChatTemplate.ExpandGemma3ImageTokens(inputTokens,
                    startId, endId, nlnlId, padId, proc.TokensPerImage);

                float[] pixels = proc.ProcessImage(msg.ImagePaths[0]);
                var emb = g3.VisionEncoder.Encode(pixels);

                int tokenStart = -1;
                for (int i = 0; i < inputTokens.Count; i++)
                {
                    if (inputTokens[i] == startId && i + 1 < inputTokens.Count && inputTokens[i + 1] == padId)
                    { tokenStart = i + 1; break; }
                }
                if (tokenStart >= 0) g3.SetVisionEmbeddings(emb, tokenStart);
                else emb.Dispose();
            }
            else if (_model is Qwen35Model q35 && q35.VisionEncoder != null)
            {
                int imagePadId = _model.Tokenizer.LookupToken("<|image_pad|>");
                if (imagePadId >= 0)
                {
                    var processor = new Qwen35ImageProcessor(q35.VisionEncoder.PatchSize, q35.VisionEncoder.SpatialMergeSize);
                    var tokenCounts = new int[msg.ImagePaths.Count];
                    for (int i = 0; i < msg.ImagePaths.Count; i++)
                    {
                        var (w, h) = Qwen35ImageProcessor.ReadPngDimensions(msg.ImagePaths[i]);
                        tokenCounts[i] = processor.ComputeImageTokenCount(h, w);
                    }
                    inputTokens = ChatTemplate.ExpandImageTokens(inputTokens, imagePadId, tokenCounts);

                    var (px, rH, rW) = processor.ProcessImage(msg.ImagePaths[0]);
                    var emb = q35.VisionEncoder.Encode(px, rH, rW);
                    int tokenStart = inputTokens.IndexOf(imagePadId);
                    if (tokenStart >= 0) q35.SetVisionEmbeddings(emb, tokenStart);
                    else emb.Dispose();
                }
            }
            return inputTokens;
        }

        private List<int> ProcessAudio(ChatMessage msg, List<int> inputTokens, string arch)
        {
            if (arch != "gemma4" || _model is not Gemma4Model g4a || g4a.AudioEncoder == null)
                return inputTokens;

            int audioStartId = _model.Tokenizer.LookupToken("<|audio>");
            int audioEndId = _model.Tokenizer.LookupToken("<audio|>");

            float[] samples = Gemma4AudioPreprocessor.DecodeAudioFile(msg.AudioPaths[0]);
            if (samples.Length % 128 != 0)
            {
                int padded = samples.Length + (128 - samples.Length % 128);
                Array.Resize(ref samples, padded);
            }

            var (melData, numFrames) = Gemma4AudioPreprocessor.ComputeMelSpectrogram(samples);
            if (melData == null || numFrames == 0) return inputTokens;

            var audioEmb = g4a.AudioEncoder.Encode(melData, numFrames);
            int numAudioTokens = (int)audioEmb.Sizes[0];

            int audioPos = -1;
            for (int i = 0; i < inputTokens.Count; i++)
            {
                if (inputTokens[i] == audioStartId) { audioPos = i; break; }
            }

            if (audioPos >= 0)
            {
                var expanded = new List<int>();
                for (int i = 0; i < audioPos; i++) expanded.Add(inputTokens[i]);
                expanded.Add(audioStartId);
                for (int i = 0; i < numAudioTokens; i++) expanded.Add(0);
                expanded.Add(audioEndId);
                for (int i = audioPos + 1; i < inputTokens.Count; i++) expanded.Add(inputTokens[i]);
                inputTokens = expanded;
                g4a.SetAudioEmbeddings(audioEmb, audioPos + 1);
            }
            else
            {
                audioEmb.Dispose();
            }

            return inputTokens;
        }

        public List<string> ScanModels(string directory)
        {
            if (!Directory.Exists(directory)) return new List<string>();
            return Directory.GetFiles(directory, "*.gguf")
                .Select(Path.GetFileName)
                .OrderBy(f => f)
                .ToList();
        }

        public void Dispose()
        {
            _model?.Dispose();
            _model = null;
            _inferenceLock.Dispose();
        }
    }
}
