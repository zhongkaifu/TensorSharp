// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using TensorSharp.Runtime.Scheduling.PrefixCache;

namespace InferenceWeb.Tests.PrefixCache;

/// <summary>§5.3.4: each of the conditions (a)-(f) failing alone.</summary>
public class DonationDecisionTests
{
    private static (PrefixTree Tree, int Scope, RadixNode X, KeyRope Key) Setup(EndStateSupport support = EndStateSupport.CopyAndDonate)
    {
        PrefixTree t = Tk.Tree(Tk.Caps(endState: support, truncation: TruncationKind.Any, pages: PageSupport.None));
        int s = Tk.Scope(t);
        KeyRope key = Tk.Key(t, Tk.Seq(1, 80));
        RadixNode x = Tk.Put(t, key, 40, s, p: 8);
        return (t, s, x, key);
    }

    private static MatchPlan PlanFor(PrefixTree t, int s, int diverge = 40, IWaitingPlanView? waiting = null)
    {
        KeyRope key = Tk.Key(t, Tk.Cat(Tk.Seq(1, diverge), Tk.Seq(900, 5)));
        return Tk.Plan(t, Tk.Req(key, s, p: 8), waiting);
    }

    [Fact]
    public void AllConditionsHold_Donates()
    {
        (PrefixTree t, int s, RadixNode x, _) = Setup();
        Assert.Equal(MaterializeMode.DonateEndState, PlanFor(t, s).Mode);
        Assert.True(t.DonationConditionsHold(x, 40, primary: false, waiting: null));
    }

    [Fact]
    public void A_NotALeaf_Clones()
    {
        (PrefixTree t, int s, RadixNode x, KeyRope key) = Setup();
        Tk.Put(t, key, 60, s, p: 8);
        MatchPlan plan = PlanFor(t, s, diverge: 41);
        Assert.Same(x, plan.PayloadNode);
        Assert.Equal(MaterializeMode.CloneEndState, plan.Mode);
    }

    [Fact]
    public void B_Locked_Clones()
    {
        (PrefixTree t, int s, RadixNode x, _) = Setup();
        LockReceipt path = t.AcquirePath(x);
        Assert.Equal(MaterializeMode.CloneEndState, PlanFor(t, s).Mode);
        t.Release(ref path);
        LockReceipt state = t.AcquireState(x);
        Assert.Equal(MaterializeMode.CloneEndState, PlanFor(t, s).Mode);
        t.Release(ref state);
        t.Pin(x);
        Assert.Equal(MaterializeMode.CloneEndState, PlanFor(t, s).Mode);
        t.Unpin(x);
        Assert.Equal(MaterializeMode.DonateEndState, PlanFor(t, s).Mode);
    }

    [Fact]
    public void C_PublicNode_Clones()
    {
        PrefixTree t = Tk.Tree(Tk.Caps(truncation: TruncationKind.None, pages: PageSupport.None));
        int s = Tk.Scope(t);
        KeyRope key = Tk.Key(t, Tk.Seq(1, 80));
        RadixNode pub = Tk.Put(t, key, 40, 0, p: 40);
        MatchPlan plan = Tk.Plan(t, Tk.Req(Tk.Key(t, Tk.Seq(1, 50)), s, p: 40));
        Assert.Same(pub, plan.PayloadNode);
        Assert.Equal(MaterializeMode.CloneEndState, plan.Mode);
    }

    [Fact]
    public void D_AnotherWaiterOfTheScope_Clones()
    {
        (PrefixTree t, int s, RadixNode x, _) = Setup();
        t.NoteRequests(s, waitingDelta: 2, runningDelta: 0);
        // No waiting view: unknown → the condition fails.
        Assert.Equal(MaterializeMode.CloneEndState, PlanFor(t, s).Mode);
        // A view that says another waiter targets X.
        var view = new FixedWaitingView { Answer = false };
        Assert.Equal(MaterializeMode.CloneEndState, PlanFor(t, s, waiting: view).Mode);
        Assert.True(view.Calls > 0);
        view.Answer = true;
        Assert.Equal(MaterializeMode.DonateEndState, PlanFor(t, s, waiting: view).Mode);

        // The helper implements "every other waiter has a current plan whose payload is not X".
        var other = new MatchPlan { Version = t.Version };
        Assert.True(PrefixTree.OtherWaitersClear(new MatchPlan?[] { other }, 1, x, t.Version));
        other.PayloadNode = x;
        Assert.False(PrefixTree.OtherWaitersClear(new MatchPlan?[] { other }, 1, x, t.Version));
        other.PayloadNode = null;
        other.Version = t.Version - 1;
        Assert.False(PrefixTree.OtherWaitersClear(new MatchPlan?[] { other }, 1, x, t.Version));
        Assert.False(PrefixTree.OtherWaitersClear(new MatchPlan?[] { null }, 1, x, t.Version));
        Assert.False(PrefixTree.OtherWaitersClear(new MatchPlan?[0], 1, x, t.Version));
        Assert.False(PrefixTree.OtherWaitersClear(new MatchPlan?[200], 129, x, t.Version));
        // More than 128 other waiters: never scanned.
        t.NoteRequests(s, waitingDelta: 200, runningDelta: 0);
        Assert.Equal(MaterializeMode.CloneEndState, PlanFor(t, s, waiting: view).Mode);
    }

    [Fact]
    public void E_NoEndStateSupport_IsRejected()
    {
        // An end state attached to a family that declares none: never donated nor cloned.
        (PrefixTree t, int s, RadixNode x, _) = Setup(EndStateSupport.None);
        MatchPlan plan = PlanFor(t, s);
        Assert.Equal(CandidateKind.None, plan.Kind);
        Assert.Equal(SourceDecline.DonateOnlyShared, plan.EndStateDecline);
        Assert.Equal(MaterializeMode.None, t.DonationDecision(x, 40, primary: false, waiting: null));
    }

    [Fact]
    public void F_TruncationBeyondTheSlack_Clones()
    {
        (PrefixTree t, int s, RadixNode x, _) = Setup();
        MatchPlan plan = PlanFor(t, s, diverge: 30);   // rewind 10 ≤ 16 → donate
        Assert.Equal(MaterializeMode.DonateEndState, plan.Mode);
        Assert.False(t.DonationConditionsHold(x, 23, primary: false, waiting: null));   // 17 > 16
        Assert.Equal(MaterializeMode.CloneEndState, t.DonationDecision(x, 23, primary: false, waiting: null));
    }

    [Fact]
    public void DonateOnly_SharedPayload_IsRejected()
    {
        (PrefixTree t, int s, RadixNode x, KeyRope key) = Setup(EndStateSupport.DonateOnly);
        Assert.Equal(MaterializeMode.DonateEndState, PlanFor(t, s).Mode);
        Tk.Put(t, key, 60, s, p: 8);                                  // (a) fails now
        MatchPlan plan = PlanFor(t, s, diverge: 41);
        Assert.Equal(CandidateKind.None, plan.Kind);
        Assert.Equal(SourceDecline.DonateOnlyShared, plan.EndStateDecline);
    }

    [Fact]
    public void PrimaryResident_KeepPrimary_Or_ConvertThenClone_Or_Rejected()
    {
        PrefixTree t = Tk.Tree(Tk.Caps(truncation: TruncationKind.None, pages: PageSupport.None, adoptPrimary: true));
        int s = Tk.Scope(t);
        KeyRope key = Tk.Key(t, Tk.Seq(1, 80));
        RadixNode p = Tk.Put(t, key, 40, s, payload: Tk.Primary(t));
        MatchPlan plan = Tk.Plan(t, Tk.Req(Tk.Key(t, Tk.Seq(1, 50)), s));
        Assert.Equal(MaterializeMode.KeepPrimary, plan.Mode);
        // Shared (a child) → convert under a tree key, then clone.
        Tk.Put(t, key, 60, s);
        plan = Tk.Plan(t, Tk.Req(Tk.Key(t, Tk.Cat(Tk.Seq(1, 41), Tk.Seq(900, 3))), s));
        Assert.Same(p, plan.PayloadNode);
        Assert.Equal(MaterializeMode.ConvertPrimaryThenClone, plan.Mode);

        // Without AdoptPrimaryOnDisplacement the shared primary is rejected.
        PrefixTree t2 = Tk.Tree(Tk.Caps(truncation: TruncationKind.None, pages: PageSupport.None, adoptPrimary: false));
        int s2 = Tk.Scope(t2);
        KeyRope key2 = Tk.Key(t2, Tk.Seq(1, 80));
        Tk.Put(t2, key2, 40, s2, payload: Tk.Primary(t2));
        Tk.Put(t2, key2, 60, s2);
        plan = Tk.Plan(t2, Tk.Req(Tk.Key(t2, Tk.Cat(Tk.Seq(1, 41), Tk.Seq(900, 3))), s2));
        Assert.Equal(CandidateKind.None, plan.Kind);
        Assert.Equal(SourceDecline.DonateOnlyShared, plan.PrimaryDecline);

        // A family with pages only (class P parity) keeps the primary: (e) reads Caps.PrimaryResident.
        PrefixTree t3 = Tk.Tree(Tk.Caps(endState: EndStateSupport.None, truncation: TruncationKind.None, pages: PageSupport.None, primaryResident: true));
        int s3 = Tk.Scope(t3);
        Tk.Put(t3, Tk.Key(t3, Tk.Seq(1, 80)), 40, s3, payload: Tk.Primary(t3));
        plan = Tk.Plan(t3, Tk.Req(Tk.Key(t3, Tk.Seq(1, 50)), s3));
        Assert.Equal(MaterializeMode.KeepPrimary, plan.Mode);
        PrefixTree t4 = Tk.Tree(Tk.Caps(endState: EndStateSupport.CopyAndDonate, truncation: TruncationKind.None, pages: PageSupport.None, primaryResident: false));
        int s4 = Tk.Scope(t4);
        Tk.Put(t4, Tk.Key(t4, Tk.Seq(1, 80)), 40, s4, payload: Tk.Primary(t4));
        plan = Tk.Plan(t4, Tk.Req(Tk.Key(t4, Tk.Seq(1, 50)), s4));
        Assert.Equal(MaterializeMode.ConvertPrimaryThenClone, plan.Mode);
    }
}
