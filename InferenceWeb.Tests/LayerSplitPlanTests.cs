using TensorSharp;

namespace InferenceWeb.Tests;

public sealed class LayerSplitPlanTests
{
    [Fact]
    public void MistralPreservesItsExistingCompiledConstructorContract()
    {
        Assert.NotNull(typeof(TensorSharp.Models.Mistral3Model).GetConstructor(
            [typeof(string), typeof(BackendType), typeof(int), typeof(ITensorParallelGroup)]));
    }

    [Fact]
    public void WeightedPartitionBalancesUnevenLayersAndEndpointWeights()
    {
        Assert.Equal(new[] { 0, 6, 8 }, LayerSplitPlan.Partition([1, 1, 1, 1, 6, 6, 6, 6], 2, 2));
        Assert.Equal(new[] { 0, 3, 10 }, LayerSplitPlan.Partition(Enumerable.Repeat(10L, 10).ToArray(), 2, 2, 40));
        Assert.Equal(new[] { 0, 7, 10 }, LayerSplitPlan.Partition(Enumerable.Repeat(10L, 10).ToArray(), 2, 2, lastBytes: 40));
        Assert.Equal(new[] { 0, 2, 4, 9 }, LayerSplitPlan.Partition(new long[9], 3, 2));
    }

    [Fact]
    public void WeightedPartitionMatchesExhaustiveOptimalCost()
    {
        var random = new Random(1017);
        for (int length = 1; length <= 12; length++)
        for (int stages = 1; stages <= Math.Min(4, length); stages++)
        for (int minimum = 1; minimum <= length / stages; minimum++)
        for (int sample = 0; sample < 8; sample++)
        {
            long[] bytes = Enumerable.Range(0, length).Select(_ => (long)random.Next(0, 101)).ToArray();
            long first = random.Next(0, 201), last = random.Next(0, 201);
            var cuts = LayerSplitPlan.Partition(bytes, stages, minimum, first, last);
            Assert.Equal(0, cuts[0]);
            Assert.Equal(length, cuts[^1]);
            long actual = 0;
            for (int stage = 0; stage < stages; stage++)
            {
                Assert.True(cuts[stage + 1] - cuts[stage] >= minimum);
                long cost = bytes.Skip(cuts[stage]).Take(cuts[stage + 1] - cuts[stage]).Sum();
                if (stage == 0) cost += first;
                if (stage == stages - 1) cost += last;
                actual = Math.Max(actual, cost);
            }
            long Enumerate(int stage, int begin)
            {
                long optimal = long.MaxValue;
                int finalEnd = length - (stages - stage - 1) * minimum;
                for (int end = begin + minimum; end <= finalEnd; end++)
                {
                    if (stage == stages - 1 && end != length) continue;
                    long cost = bytes.Skip(begin).Take(end - begin).Sum();
                    if (stage == 0) cost += first;
                    if (stage == stages - 1) cost += last;
                    else cost = Math.Max(cost, Enumerate(stage + 1, end));
                    optimal = Math.Min(optimal, cost);
                }
                return optimal;
            }
            Assert.Equal(Enumerate(0, 0), actual);
        }
    }

    [Fact]
    public void WeightedPlansOwnEveryLayerOnceAndUseEveryDevice()
    {
        var random = new Random(41);
        for (int nodes = 1; nodes <= 4; nodes++)
        for (int devices = 1; devices <= 4; devices++)
        {
            long[] bytes = Enumerable.Range(0, nodes * devices + 9).Select(_ => (long)random.Next(0, 1000)).ToArray();
            var owners = new int[bytes.Length];
            for (int node = 0; node < nodes; node++)
            {
                var plan = LayerSplitPlan.CreateWeighted(bytes, devices, node, nodes, 4000, 1000);
                var perDevice = new int[devices];
                for (int layer = plan.FirstLayer; layer < plan.EndLayer; layer++)
                {
                    owners[layer]++;
                    perDevice[plan.LocalRank(layer)]++;
                }
                Assert.All(perDevice, count => Assert.True(count > 0));
            }
            Assert.All(owners, count => Assert.Equal(1, count));
        }
    }

    [Theory]
    [InlineData(2, 1)]
    [InlineData(2, 3)]
    [InlineData(3, 2)]
    [InlineData(4, 1)]
    public void DriverLastOrderRunsTheSameStagesOnRotatedNodes(int nodes, int devices)
    {
        long[] bytes = Enumerable.Range(0, 40).Select(i => (long)(i * 37 % 11 + 1)).ToArray();
        for (int node = 0; node < nodes; node++)
        {
            var rotated = LayerSplitPlan.CreateWeighted(bytes, devices, node, nodes, 50, 70, true);
            int stage = (node + nodes - 1) % nodes;
            // A stage owns the same layers and device split whichever node runs it.
            var reference = LayerSplitPlan.CreateWeighted(bytes, devices, stage, nodes, 50, 70);
            Assert.Equal(stage, rotated.StageIndex);
            Assert.Equal(reference.FirstLayer, rotated.FirstLayer);
            Assert.Equal(reference.EndLayer, rotated.EndLayer);
            for (int layer = rotated.FirstLayer; layer < rotated.EndLayer; layer++)
                Assert.Equal(reference.LocalRank(layer), rotated.LocalRank(layer));
            Assert.Equal(stage == 0, rotated.IsFirst);
            Assert.Equal(stage == nodes - 1, rotated.IsLast);
            Assert.Equal(node == 0, rotated.IsLast);
            if (!rotated.IsFirst) Assert.Equal((node + nodes - 1) % nodes, rotated.PreviousNode);
            if (!rotated.IsLast) Assert.Equal((node + 1) % nodes, rotated.NextNode);
        }
        Assert.Equal(1, LayerSplitPlan.NodeOfStage(0, nodes, true));
        Assert.Equal(0, LayerSplitPlan.NodeOfStage(nodes - 1, nodes, true));
        Assert.Equal(nodes - 1, LayerSplitPlan.StageOf(0, nodes, true));
    }

    [Fact]
    public void LegacyOrderAndSingleNodePlansAreUnchanged()
    {
        var legacy = new LayerSplitPlan(12, 2, 1, 3);
        Assert.False(legacy.DriverRunsLastStage);
        Assert.Equal(1, legacy.StageIndex);
        Assert.Equal(0, legacy.PreviousNode);
        Assert.Equal(2, legacy.NextNode);
        var single = new LayerSplitPlan(12, 3, 0, 1, true);
        Assert.False(single.DriverRunsLastStage);
        Assert.True(single.IsFirst && single.IsLast);
    }

    [Fact]
    public void PartitionRejectsInvalidCostsAndPreservesMaximumRepresentableCost()
    {
        Assert.Throws<ArgumentException>(() => LayerSplitPlan.Partition([1, 2, 3], 2, 2));
        Assert.Throws<ArgumentOutOfRangeException>(() => LayerSplitPlan.Partition([1, -1], 2, 1));
        Assert.Throws<ArgumentOutOfRangeException>(() => LayerSplitPlan.Partition([1], 1, 1, -1));
        Assert.Throws<OverflowException>(() => LayerSplitPlan.Partition([long.MaxValue, 1], 1, 1));
        Assert.Equal(new[] { 0, 2 }, LayerSplitPlan.Partition([long.MaxValue, 0], 1, 1));
        Assert.Equal(new[] { 0, 1, 2 }, LayerSplitPlan.Partition([0, long.MaxValue], 2, 1));
    }

    [Fact]
    public void CapacityPartitionGivesLayersInProportionToCapacity()
    {
        long[] even = Enumerable.Repeat(10L, 12).ToArray();
        // Equal capacities: the equal-cost partition.
        Assert.Equal(LayerSplitPlan.Partition(even, 3, 1), LayerSplitPlan.PartitionByCapacity(even, [1, 1, 1], [1, 1, 1]));
        Assert.Equal(new[] { 0, 8, 12 }, LayerSplitPlan.PartitionByCapacity(even, [2, 1], [1, 1]));
        Assert.Equal(new[] { 0, 4, 12 }, LayerSplitPlan.PartitionByCapacity(even, [1, 2], [1, 1]));
        Assert.Equal(new[] { 0, 2, 6, 12 }, LayerSplitPlan.PartitionByCapacity(even, [1, 2, 3], [1, 1, 1]));
        // A minimum per part (a node's stage count) binds before capacity does.
        Assert.Equal(new[] { 0, 9, 12 }, LayerSplitPlan.PartitionByCapacity(even, [100, 1], [1, 3]));
        // Endpoint bytes count against the first and last parts.
        Assert.Equal(new[] { 0, 5, 12 }, LayerSplitPlan.PartitionByCapacity(even, [1, 1], [1, 1], firstBytes: 20));
        Assert.Throws<ArgumentException>(() => LayerSplitPlan.PartitionByCapacity(even, [1, 1], [6, 7]));
        Assert.Throws<ArgumentOutOfRangeException>(() => LayerSplitPlan.PartitionByCapacity(even, [1, 0], [1, 1]));
    }

    [Fact]
    public void CapacityPartitionMatchesExhaustiveOptimalRatio()
    {
        var random = new Random(1112);
        for (int length = 2; length <= 10; length++)
        for (int parts = 1; parts <= Math.Min(3, length); parts++)
        for (int sample = 0; sample < 10; sample++)
        {
            long[] bytes = Enumerable.Range(0, length).Select(_ => (long)random.Next(0, 100)).ToArray();
            long[] capacity = Enumerable.Range(0, parts).Select(_ => (long)random.Next(1, 5)).ToArray();
            int[] minimum = Enumerable.Repeat(1, parts).ToArray();
            int[] cuts = LayerSplitPlan.PartitionByCapacity(bytes, capacity, minimum);
            Assert.Equal(Worst(bytes, capacity, cuts), BestRatio(bytes, capacity, 0, 0), 12);
        }

        static double Worst(long[] bytes, long[] capacity, int[] cuts)
            => Enumerable.Range(0, capacity.Length).Max(p => bytes[cuts[p]..cuts[p + 1]].Sum() / (double)capacity[p]);

        static double BestRatio(long[] bytes, long[] capacity, int part, int begin)
        {
            if (part == capacity.Length - 1) return bytes[begin..].Sum() / (double)capacity[part];
            double best = double.MaxValue;
            for (int end = begin + 1; end <= bytes.Length - (capacity.Length - part - 1); end++)
                best = Math.Min(best, Math.Max(bytes[begin..end].Sum() / (double)capacity[part], BestRatio(bytes, capacity, part + 1, end)));
            return best;
        }
    }

    [Fact]
    public void NodesOfDifferentShapesAgreeOnOnePlan()
    {
        // Node 0 (the driver, which runs the last pipeline position): one stage of two GPUs with 96 GiB; node 1: two
        // single-GPU stages of 24 GiB each; node 2: one 24 GiB GPU.
        var topology = new LayerSplitTopology([1, 2, 1], [2, 1, 1], [96, 24, 24]);
        Assert.False(topology.IsUniform);
        Assert.Equal(4, topology.TotalStages);
        Assert.Equal(5, topology.TotalDevices);
        Assert.Equal(new[] { 0, 2, 4 }, Enumerable.Range(0, 3).Select(topology.DeviceOffset));
        var plans = Enumerable.Range(0, 3).Select(node => LayerSplitPlan.Create(48, null, topology, node, 0, 0, driverRunsLastStage: true)).ToArray();
        // Pipeline order: node 1, node 2, node 0. Capacities 48 : 24 : 96 of 168 GiB -> 13.7, 6.9, 27.4 layers; the
        // smallest worst ratio is 7/24 (= 14/48 = 28/96), reached first (smallest boundaries) by 13, 7 and 28 layers.
        Assert.Equal((0, 13), (plans[1].FirstLayer, plans[1].EndLayer));
        Assert.Equal((13, 20), (plans[2].FirstLayer, plans[2].EndLayer));
        Assert.Equal((20, 48), (plans[0].FirstLayer, plans[0].EndLayer));
        Assert.All(plans, plan => Assert.Equal(4, plan.GlobalDegree));
        // Node 1 splits its 13 layers over its two stages as the uniform plan would (6 + 7).
        Assert.Equal(0, plans[1].LocalRank(5));
        Assert.Equal(1, plans[1].LocalRank(6));
        Assert.Equal(0, plans[0].LocalRank(47));
        // A uniform topology is the existing plan.
        var uniform = new LayerSplitTopology([2, 2], [1, 1], [24, 24]);
        var existing = new LayerSplitPlan(10, 2, 1, 2, true);
        var agreed = LayerSplitPlan.Create(10, null, uniform, 1, 0, 0, true);
        Assert.Equal((existing.FirstLayer, existing.EndLayer, existing.LocalRank(existing.FirstLayer + 2)),
            (agreed.FirstLayer, agreed.EndLayer, agreed.LocalRank(agreed.FirstLayer + 2)));
        Assert.Throws<NotSupportedException>(() => LayerSplitPlan.Create(3, null, topology, 0, 0, 0, true));
    }
}
