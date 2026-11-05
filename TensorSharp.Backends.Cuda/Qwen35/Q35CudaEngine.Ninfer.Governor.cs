// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
namespace TensorSharp.Cuda;

public sealed unsafe partial class Q35CudaEngine
{
    /// <summary>Backend-only capability for the measured fixed native-draft
    /// profile. Model ownership, resolved drafter and sampler are checked above.</summary>
    public bool NinferFixedDraftGovernorQualified => _m?.Ninfer != null
        && FixedDraftGovernorGeometry(_ninferQualifiedDefaults?.Qualified == true, _sms,
            _ninferVerificationEnabled, HasNinferDrafter,
            _ninferDraftDesc?.HeadPolicy == CudaNinferDraftHeadPolicy.FullTargetVocabulary,
            _ninferDraftDesc?.DraftTokens ?? 0, _m.NCtx,
            NinferDecodeGraphsEnabled, HasQualifiedGovernorVector(NinferA16Selection, NinferA16ModulePath),
            NinferFlashInferEnabled, NinferExactVerifyEnabled, NinferDraftUsesShortlist);

    internal static bool HasQualifiedGovernorVector(string selection, string path)
        => selection is "auto-sm120" or "explicit-legacy-exact"
            && System.IO.Path.GetFileNameWithoutExtension(path) == "tensorsharp_q35_ninfer_a16_legacy_exact_vector";

    internal static bool FixedDraftGovernorGeometry(bool qualifiedRecipe, int sms, bool verification,
        bool nativeDraft, bool fullHead, int draftTokens, int context,
        bool graphs, bool legacyVector, bool flashInfer, bool exactVerify, bool shortlist)
        => qualifiedRecipe && sms == 170 && verification && nativeDraft && fullHead && draftTokens == 7
            && context == 40960 && graphs && legacyVector && !flashInfer && !exactVerify && !shortlist;
}
