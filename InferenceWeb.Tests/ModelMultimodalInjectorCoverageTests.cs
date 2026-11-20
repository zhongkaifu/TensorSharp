// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System.Reflection;

using TensorSharp.Models.Architecture;

namespace InferenceWeb.Tests;

/// <summary>
/// Structural guard for the multimodal hand-off.
///
/// The failure this exists for: a vision model whose encoder runs and whose prompt
/// grows by N image rows, but which is never handed the embeddings - so those rows keep
/// the embedding of the filler token and the model answers as if it never saw the
/// image. That is exactly what happened to Muse-Glimmer: <c>LoadProjectors</c> and its
/// prompt expansion were both wired up, but neither queue stage had a
/// <c>MuseGlimmerModel</c> case. Asked "这是什么图片" about a banner, the model replied
/// that it had been given "a huge block of characters, all 'o' repeated".
///
/// The injector used to route every stage through its own <c>switch (_model)</c> over
/// concrete model types, so a new model had to be added to FOUR of them and missing one
/// failed silently; this file used to scan the IL of those switches and compare their
/// coverage. Those switches are gone: loading and queueing now go through
/// <see cref="IVisionCapableModel"/> / <see cref="IAudioCapableModel"/>, which a model
/// implements once and cannot implement by halves. What remains to guard is the seam
/// between the capability interfaces themselves - a model that expands image
/// placeholders but forgot to declare that it can receive the embeddings would fail the
/// same silent way.
/// </summary>
public class ModelMultimodalInjectorCoverageTests
{
    private static readonly BindingFlags AnyMethod =
        BindingFlags.Instance | BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic;

    private static readonly Type PromptExpander =
        typeof(ModelBase).Assembly.GetType("TensorSharp.Models.Architecture.IMultimodalPromptExpander")
        ?? throw new InvalidOperationException("IMultimodalPromptExpander not found — did the seam get renamed?");

    private static IEnumerable<Type> ConcreteModels =>
        typeof(ModelBase).Assembly.GetTypes()
            .Where(t => typeof(ModelBase).IsAssignableFrom(t) && !t.IsAbstract);

    /// <summary>
    /// A model that expands its own media placeholders MUST also declare that it can
    /// receive the embeddings those placeholders stand for. Expanding without receiving
    /// is the silent-image bug.
    /// </summary>
    [Fact]
    public void EveryModelThatExpandsImagePlaceholders_CanAlsoReceiveTheEmbeddings()
    {
        var expanders = ConcreteModels.Where(t => PromptExpander.IsAssignableFrom(t)).ToList();
        Assert.NotEmpty(expanders);

        var missing = expanders
            .Where(t => !typeof(IVisionCapableModel).IsAssignableFrom(t)
                        && !typeof(IAudioCapableModel).IsAssignableFrom(t))
            .Select(t => t.Name)
            .OrderBy(n => n)
            .ToList();

        Assert.True(missing.Count == 0,
            $"These models expand media placeholders but declare no way to receive embeddings: " +
            $"{string.Join(", ", missing)}. They would run their encoder and then silently drop the " +
            "result, leaving the image rows as filler-token embeddings.");
    }

    /// <summary>
    /// The converse, and the reason the old IL scan existed: a model that CAN receive
    /// vision embeddings (it has the method) must declare the interface, or the injector
    /// - which now only ever sees the interface - will never call it.
    /// </summary>
    [Fact]
    public void EveryModelWithAVisionHook_DeclaresTheVisionInterface()
    {
        var undeclared = ConcreteModels
            .Where(t => t.GetMethod("SetVisionEmbeddings", AnyMethod) != null)
            .Where(t => !typeof(IVisionCapableModel).IsAssignableFrom(t))
            .Select(t => t.Name)
            .OrderBy(n => n)
            .ToList();

        Assert.True(undeclared.Count == 0,
            $"These models have a SetVisionEmbeddings hook but do not implement IVisionCapableModel, " +
            $"so nothing will ever call it: {string.Join(", ", undeclared)}.");
    }

    /// <summary>Same, for the audio hand-off.</summary>
    [Fact]
    public void EveryModelWithAnAudioHook_DeclaresTheAudioInterface()
    {
        var undeclared = ConcreteModels
            .Where(t => t.GetMethod("SetAudioEmbeddings", AnyMethod) != null)
            .Where(t => !typeof(IAudioCapableModel).IsAssignableFrom(t))
            .Select(t => t.Name)
            .OrderBy(n => n)
            .ToList();

        Assert.True(undeclared.Count == 0,
            $"These models have a SetAudioEmbeddings hook but do not implement IAudioCapableModel: " +
            $"{string.Join(", ", undeclared)}.");
    }

    /// <summary>A vision tower nobody can load is a tower nobody can use.</summary>
    [Fact]
    public void EveryVisionCapableModel_CanLoadItsProjector()
    {
        foreach (Type model in ConcreteModels.Where(t => typeof(IVisionCapableModel).IsAssignableFrom(t)))
        {
            Assert.True(model.GetMethod("LoadVisionEncoder", AnyMethod) != null,
                $"{model.Name} implements IVisionCapableModel but has no LoadVisionEncoder.");
        }
    }

    /// <summary>Muse-Glimmer specifically — the model this guard was written for.</summary>
    [Fact]
    public void MuseGlimmer_DeclaresTheWholeVisionContract()
    {
        Assert.True(typeof(IVisionCapableModel).IsAssignableFrom(typeof(MuseGlimmerModel)),
            "MuseGlimmerModel does not declare IVisionCapableModel, so its images are silently dropped.");
        Assert.True(PromptExpander.IsAssignableFrom(typeof(MuseGlimmerModel)),
            "MuseGlimmerModel does not declare its prompt expansion, so its image tokens never expand.");
        // The unified-runtime family (stage S4 of docs/design/muse-glimmer-unified.md) carries the same contract.
        Assert.True(typeof(IVisionCapableModel).IsAssignableFrom(typeof(MuseGlimmerRuntimeModel)),
            "MuseGlimmerRuntimeModel does not declare IVisionCapableModel, so its images are silently dropped.");
        Assert.True(PromptExpander.IsAssignableFrom(typeof(MuseGlimmerRuntimeModel)),
            "MuseGlimmerRuntimeModel does not declare its prompt expansion, so its image tokens never expand.");
    }
}
