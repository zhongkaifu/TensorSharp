// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.IO;

namespace TensorSharp.Cuda
{
    public enum CudaW8HeadScope { TargetAndDraft, DraftOnly }

    /// <summary>Explicit experimental companion output head, independent of ggml weight types.
    /// The engine synchronously copies the complete original payload during construction;
    /// the caller may close Source after that constructor returns, including on failure.
    /// This does not declare support for a complete NInfer model. DraftOnly retains
    /// the original Q6 target head; the shared scope replaces target head math too.</summary>
    public sealed class CudaW8HeadWeightDesc
    {
        public const string SharedProfile = "ninfer-w8g32-f16s-bf16-companion-v1";
        public const string DraftProfile = "ninfer-w8g32-f16s-bf16-draft-companion-v1";
        public ICudaWeightSource Source { get; }
        public long SourceOffset { get; }
        public long Bytes { get; }
        public int Rows { get; }
        public int Columns { get; }
        public string ModulePath { get; }
        public string Provenance { get; }
        public string Profile { get; }
        public string ExpectedPayloadSha256 { get; }
        public CudaW8HeadScope Scope { get; }

        public CudaW8HeadWeightDesc(ICudaWeightSource source, long sourceOffset, long bytes,
            int rows, int columns, string modulePath, string provenance,
            string profile = SharedProfile, string expectedPayloadSha256 = null,
            CudaW8HeadScope scope = CudaW8HeadScope.TargetAndDraft)
        {
            ArgumentNullException.ThrowIfNull(source);
            if (scope != CudaW8HeadScope.TargetAndDraft && scope != CudaW8HeadScope.DraftOnly)
                throw new ArgumentOutOfRangeException(nameof(scope));
            if (sourceOffset < 0 || bytes != CudaW8G32HeadModule.WeightBytes
                || sourceOffset > long.MaxValue - bytes || rows != CudaW8G32HeadModule.Vocabulary
                || columns != CudaW8G32HeadModule.Width)
                throw new ArgumentException("The companion requires the exact original W8G32_F16S 248320x5120 payload and a bounded source range.");
            if (string.IsNullOrWhiteSpace(modulePath) || !Path.IsPathFullyQualified(modulePath)
                || Path.GetFullPath(modulePath) != modulePath)
                throw new ArgumentException("An explicit normalized absolute W8 module path is required.", nameof(modulePath));
            if (string.IsNullOrEmpty(profile) || profile.Length > 128)
                throw new ArgumentException("A bounded companion profile is required.", nameof(profile));
            foreach (char c in profile)
                if (!char.IsAsciiLetterOrDigit(c) && c != '-' && c != '_' && c != '.')
                    throw new ArgumentException("The companion profile contains an unsupported character.", nameof(profile));
            if (string.IsNullOrWhiteSpace(provenance) || provenance.Length > 8192)
                throw new ArgumentException("Explicit source provenance is required.", nameof(provenance));
            if (expectedPayloadSha256 != null)
            {
                if (expectedPayloadSha256.Length != 64)
                    throw new ArgumentException("Expected payload SHA256 must have 64 hexadecimal digits.", nameof(expectedPayloadSha256));
                foreach (char c in expectedPayloadSha256)
                    if (!Uri.IsHexDigit(c)) throw new ArgumentException("Expected payload SHA256 is invalid.", nameof(expectedPayloadSha256));
            }
            Source = source; SourceOffset = sourceOffset; Bytes = bytes; Rows = rows; Columns = columns;
            ModulePath = modulePath; Provenance = provenance;
            Profile = scope == CudaW8HeadScope.DraftOnly && profile == SharedProfile ? DraftProfile : profile;
            Scope = scope;
            ExpectedPayloadSha256 = expectedPayloadSha256?.ToLowerInvariant();
        }
    }
}
