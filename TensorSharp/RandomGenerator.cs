// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
﻿using System;

namespace TensorSharp
{
    public class RandomGenerator
    {
        static Random rnd = new Random(DateTime.Now.Millisecond);

        public int NextSeed()
        {
            return rnd.Next();
        }

        public static float[] BuildRandomUniformWeight(long[] sizes, float min, float max)
        {
            long size = 1;
            foreach (var s in sizes)
            {
                size *= s;
            }

            float[] w = new float[size];

            for (int i = 0; i < size; i++)
            {
                w[i] = (float)rnd.NextDouble() * (max - min) + min;
            }

            return w;
        }


        public static float[] BuildRandomBernoulliWeight(long[] sizes, float p)
        {
            long size = 1;
            foreach (var s in sizes)
            {
                size *= s;
            }

            float[] w = new float[size];
            

            for (int i = 0; i < size; i++)
            {
                w[i] = rnd.NextDouble() <= p ? 1.0f : 0.0f;
            }

            return w;
        }
    }
}
